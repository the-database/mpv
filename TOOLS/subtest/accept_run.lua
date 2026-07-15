-- accept_run.lua -- scripted acceptance timing run (WP-H7).
--
-- Reproduces the kit's manual run_tests.bat procedure without the human:
-- play from --start, perform the mid-run seek that exercises the
-- render-ahead flush/refill path (the .bat said "PRESS RIGHT-ARROW ONCE
-- ~2 MIN IN"; this issues the same `seek 5 relative` on a timer), quit after
-- the configured duration. --length is NOT used so the seek cannot shift the
-- end-of-window; wall-clock duration is enforced here.
--
-- Options (--script-opts=arun-<name>=<value>):
--   duration  total wall seconds to play (default 360)
--   seek_at   wall seconds in at which to seek (default 120; 0 disables)
--   seek_by   relative seek amount in seconds (default 5)

local opts = {
    duration = 360,
    seek_at = 120,
    seek_by = 5,
}
require("mp.options").read_options(opts, "arun")

local started = false

mp.register_event("file-loaded", function()
    if started then return end
    started = true
    mp.set_property_bool("pause", false)
    if opts.seek_at > 0 then
        mp.add_timeout(opts.seek_at, function()
            mp.msg.info("arun: mid-run seek")
            mp.commandv("seek", tostring(opts.seek_by), "relative")
        end)
    end
    mp.add_timeout(opts.duration, function()
        mp.msg.info("arun: done")
        mp.command("quit")
    end)
end)
