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
--   seek_to   ABSOLUTE media-time seek target; when > 0 it replaces the
--             relative seek (WP-H10: aim the mid-run seek INTO a known dense
--             wall -- e.g. the ep09 typeset at ~1121.4-1125.9s -- instead of
--             wherever the relative jump happens to land)

local opts = {
    duration = 360,
    seek_at = 120,
    seek_by = 5,
    seek_to = 0,
}
require("mp.options").read_options(opts, "arun")

local started = false

mp.register_event("file-loaded", function()
    if started then return end
    started = true
    mp.set_property_bool("pause", false)
    if opts.seek_at > 0 then
        mp.add_timeout(opts.seek_at, function()
            if opts.seek_to > 0 then
                mp.msg.info(string.format(
                    "arun: mid-run seek to %.1f absolute", opts.seek_to))
                mp.commandv("seek", tostring(opts.seek_to), "absolute")
            else
                mp.msg.info("arun: mid-run seek")
                mp.commandv("seek", tostring(opts.seek_by), "relative")
            end
        end)
    end
    mp.add_timeout(opts.duration, function()
        mp.msg.info("arun: done")
        mp.command("quit")
    end)
end)
