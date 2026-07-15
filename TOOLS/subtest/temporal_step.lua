-- temporal_step.lua -- frame-step screenshot driver for temporal_ab.py.
--
-- Bundled (--script=...) so it works identically for a local Linux mpv and
-- for a Windows mpv.exe driven from WSL (no IPC socket needed, which is the
-- part that does not cross the interop boundary).
--
-- Flow: file-loaded -> pause -> exact seek to tstep-start -> settle ->
-- [screenshot-to-file, frame-step, wait for the step to land, settle] x N ->
-- write "<out>/DONE" marker -> quit.
--
-- Ordering guarantee relied on: the screenshot-to-file VOCTRL is processed on
-- the VO thread strictly after the stepped frame's queued draw, so the PNG is
-- of the new frame once time-pos has advanced. The settle delay additionally
-- lets the render-ahead worker bank ahead between steps so a step is never
-- served stale (ra-stale is also asserted from the stats dump by the driver).
--
-- Options (--script-opts=tstep-<name>=<value>):
--   start   (number) seek target in seconds
--   frames  (number) how many frames to capture
--   out     (string) output dir; MUST already exist. Forward slashes work on
--                    Windows too, so the driver always passes them.
--   settle  (number) seconds to wait between the step landing and the shot
--   settle0 (number) seconds to wait after the initial seek (RA warm-up)
--   quitpause (number) extra seconds to idle before quit (flush stats)

local opts = {
    start = 0,
    frames = 1,
    out = ".",
    settle = 0.25,
    settle0 = 3.0,
    quitpause = 0.5,
}
require("mp.options").read_options(opts, "tstep")

local idx = 0
local stepping = false      -- a frame-step is in flight
local started = false
local last_pos = nil

local function shot_path(i)
    return string.format("%s/f%05d.png", opts.out, i)
end

local function finish()
    local f = io.open(opts.out .. "/DONE", "w")
    if f then f:write(string.format("frames=%d\n", idx)); f:close() end
    mp.add_timeout(opts.quitpause, function() mp.command("quit") end)
end

local function shoot_and_step()
    -- The tstep-shot line lands in the mpv log with an mp_time timestamp --
    -- the same clock --dump-stats uses -- so the driver can attribute counter
    -- deltas to the exact inter-frame interval (first-divergence forensics).
    mp.msg.info(string.format("tstep-shot idx=%d pos=%.4f",
                              idx, mp.get_property_number("time-pos") or -1))
    mp.commandv("screenshot-to-file", shot_path(idx), "window")
    idx = idx + 1
    if idx >= opts.frames then
        finish()
        return
    end
    stepping = true
    mp.commandv("frame-step")
end

-- Failsafe: if a frame-step never lands (EOF inside the window), finish with
-- what we have rather than hanging the driver forever.
local watchdog = mp.add_periodic_timer(30.0, function()
    if started and stepping then
        mp.msg.warn("tstep: step did not land in 30s; finishing early")
        finish()
    end
end)
watchdog:kill()

mp.observe_property("time-pos", "number", function(_, v)
    if v == nil then return end
    if stepping and last_pos ~= nil and v ~= last_pos then
        stepping = false
        watchdog:kill(); watchdog:resume()
        mp.add_timeout(opts.settle, shoot_and_step)
    end
    last_pos = v
end)

mp.register_event("file-loaded", function()
    if started then return end
    mp.set_property_bool("pause", true)
    mp.commandv("seek", tostring(opts.start), "absolute+exact")
end)

mp.register_event("playback-restart", function()
    if started then return end
    started = true
    watchdog:resume()
    -- settle0: let the seek/prewarm settle before the first capture
    mp.add_timeout(opts.settle0, shoot_and_step)
end)
