-- osc_hover_rig.lua -- rig-side driver for the OSC hover-zone test (WP-H9).
--
-- osc_hover_test.py drives mpv over a unix IPC socket, which does not cross
-- the WSL->mpv.exe interop boundary. This lua replicates its scenario body
-- in-process (same probes, same ordering); the offline adjudicator
-- (osc_hover_rig_check.py) applies osc_hover_test.py's band_diff_frac
-- detection to the captured screenshots.
--
-- Window size is taken from osd-dimensions AT TEST TIME (the WM may fit the
-- borderless window to the work area, so --geometry is not authoritative;
-- observed on the rig: 7680x4320 requested -> 7338x4128 granted).
-- osd-dimensions is itself under test, but the adjudicator cross-checks it
-- against the screenshot dimensions: if it were corrupted to capped space,
-- the "true band" sweep would land in the displaced hover zone and the
-- osd_dim_true=false + band signature still yields a BROKEN verdict.
--
-- Probes (mirrors osc_hover_test.Scenario.run):
--   baseline shot (no mouse entered) -> true-band sweep (y = H-20) ->
--   mouse-pos.hover + shot (x2 attempts) -> park mid-window, wait hide,
--   shot -> false-band sweep (y = 0.875*probe_cap, x < w*cap/H) -> shot.
--
-- Options (--script-opts=ohr-<name>=<value>):
--   out        output dir (forward slashes; must exist)
--   expect_w/h fallback window size if osd-dimensions is unavailable
--   probe_cap  cap used to locate the FALSE band probe (default 1440)
local opts = {
    out = ".",
    expect_w = 7680,
    expect_h = 4320,
    probe_cap = 1440,
}
require("mp.options").read_options(opts, "ohr")

local w, h = opts.expect_w, opts.expect_h

local steps = {}
local si = 0
local function next_step()
    si = si + 1
    if steps[si] then steps[si]() end
end
local function delay(t)
    return function() mp.add_timeout(t, next_step) end
end

local function shot(name)
    return function()
        mp.commandv("screenshot-to-file",
                    string.format("%s/%s.png", opts.out, name), "window")
        mp.msg.info("ohr: shot " .. name)
        mp.add_timeout(0.6, next_step)
    end
end

-- Runtime sweep: 5 in-zone positions, 0.12 s apart, then cont(). Every
-- position stays inside [0,w) x the band (leaving the showhide area in x
-- or y delivers MOUSE_LEAVE, which insta-hides the OSC).
local FXS = { 0.15, 0.35, 0.55, 0.75, 0.85 }
local function run_sweep(y, width, cont)
    local i = 0
    local function step()
        i = i + 1
        if i > #FXS then cont() return end
        mp.commandv("mouse", tostring(math.floor(width * FXS[i])),
                    tostring(math.floor(y)))
        mp.add_timeout(0.12, step)
    end
    step()
end

local function sweep_true()
    run_sweep(h - 20, w, next_step)
end
local function sweep_false()
    run_sweep(math.floor(opts.probe_cap * 0.875),
              math.floor(w * opts.probe_cap / h + 0.5), next_step)
end

local function log_probes(label)
    return function()
        local dim = mp.get_property_native("osd-dimensions") or {}
        local mpos = mp.get_property_native("mouse-pos") or {}
        mp.msg.info(string.format(
            "ohr: probe %s osd-dim=%dx%d hover=%s mouse=%s,%s",
            label, dim.w or -1, dim.h or -1, tostring(mpos.hover),
            tostring(mpos.x), tostring(mpos.y)))
        next_step()
    end
end

local function add(list)
    for _, f in ipairs(list) do steps[#steps + 1] = f end
end

add({ delay(2.0) })
add({ function()
    local dim = mp.get_property_native("osd-dimensions") or {}
    if dim.w and dim.w > 0 then w = dim.w end
    if dim.h and dim.h > 0 then h = dim.h end
    mp.msg.info(string.format("ohr: runtime dims %dx%d (expected %dx%d)",
                              w, h, opts.expect_w, opts.expect_h))
    next_step()
end })
add({ shot("baseline"), delay(1.0), shot("baseline_chk") })
add({ log_probes("baseline") })
-- TRUE band, attempt 1
add({ sweep_true })
add({ delay(0.35), log_probes("true_band"), shot("true_band") })
-- TRUE band, attempt 2 (fade-race dodge, mirrors the python retry)
add({ sweep_true })
add({ delay(0.35), shot("true_band2") })
-- reset: park between both candidate zones, wait out hidetimeout+fade
add({ function()
    mp.commandv("mouse", tostring(math.floor(w * 0.5)),
                tostring(math.floor(h * 0.40)))
    mp.add_timeout(0.12, next_step)
end })
add({ delay(4.0), shot("reset") })
-- FALSE band: the capped-space hover zone
add({ sweep_false })
add({ delay(0.35), log_probes("false_band"), shot("false_band") })
add({ function()
    local f = io.open(opts.out .. "/DONE", "w")
    if f then f:write("ok\n"); f:close() end
    mp.msg.info("ohr: done")
    mp.add_timeout(0.5, function() mp.command("quit") end)
end })

local started = false
mp.register_event("file-loaded", function()
    if started then return end
    started = true
    mp.msg.info("ohr: start")
    next_step()
end)
