-- statspage_run.lua -- scripted stats-page stress run (WP-H7 defect 2).
--
-- Round 3 proved a manual "press i" procedure is fragile (the page never got
-- opened); this drives it deterministically: play from --start, open the
-- stats page after `open_at` seconds via the stats.lua script-binding, keep
-- playing for `duration` seconds with the page up, screenshot the window
-- every `shots_every` seconds (proof the page is open AND visibly updating),
-- then quit. Drop/spike adjudication comes from --dump-stats as usual.
--
-- Options (--script-opts=sprun-<name>=<value>):
--   open_at     seconds of playback before opening the stats page (default 5)
--   duration    seconds to keep playing with the page open (default 60)
--   shots_every screenshot interval in seconds; 0 disables (default 0)
--   out         output dir for screenshots (must exist)

local opts = {
    open_at = 5,
    duration = 60,
    shots_every = 0,
    out = ".",
}
require("mp.options").read_options(opts, "sprun")

local shot_n = 0
local started = false

mp.register_event("file-loaded", function()
    if started then return end
    started = true
    mp.set_property_bool("pause", false)
    mp.add_timeout(opts.open_at, function()
        mp.commandv("script-binding", "stats/display-stats-toggle")
        mp.msg.info("sprun: stats page opened")
        if opts.shots_every > 0 then
            local t = mp.add_periodic_timer(opts.shots_every, function()
                mp.commandv("screenshot-to-file",
                            string.format("%s/sp%03d.png", opts.out, shot_n),
                            "window")
                shot_n = shot_n + 1
            end)
        end
        mp.add_timeout(opts.duration, function()
            mp.msg.info("sprun: done")
            mp.command("quit")
        end)
    end)
end)
