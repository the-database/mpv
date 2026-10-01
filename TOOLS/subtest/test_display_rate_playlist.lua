-- Hardware regression check. Start at a different desktop refresh rate, with
-- refresh matching enabled and a playlist of clips lasting at least 6 seconds.
-- For 23.976 -> 23.976 -> 29.97 use the defaults; for two equal-rate episodes,
-- add --script-opts=display_playlist_test-files=2,display_playlist_test-switches=1.
-- Use --idle=yes --force-window=yes or --keep-open=yes to check restoration
-- before the window closes. This script quits automatically after the check.
local mp = require 'mp'
local options = require 'mp.options'
local opts = {files = 3, switches = 2}
options.read_options(opts, 'display_playlist_test')

local loaded, switches, restores = 0, 0, 0
local original, finishing, failed, ending
local finish

mp.enable_messages('info')
mp.register_event('log-message', function(event)
    if not event.prefix:find('/win32', 1, true) then return end
    if event.text:find('Matched ', 1, true) then switches = switches + 1 end
    if event.text:find('Restored original display refresh rate.', 1, true) then
        restores = restores + 1
        if loaded < opts.files then failed = 'restored before the last episode' end
        finish()
    end
end)

mp.register_event('file-loaded', function()
    loaded = loaded + 1
    original = original or mp.get_property_number('display-fps')
    mp.msg.info(string.format('Episode %d started; switches=%d restores=%d',
                             loaded, switches, restores))
end)

finish = function()
    -- EOF can be delivered before the synchronous Windows restore finishes.
    -- Wait for its log event too; a property read can otherwise block while
    -- that event is still queued, producing a false zero-restores failure.
    if finishing or not ending or loaded < opts.files or restores == 0 then return end
    finishing = true
    mp.add_timeout(0.5, function()
        local current = mp.get_property_number('display-fps', 0)
        local ok = not failed and loaded == opts.files and
                   switches == opts.switches and restores == 1 and original and
                   math.abs(current - original) < 0.1
        local message = string.format('episodes=%d switches=%d restores=%d; %s',
            loaded, switches, restores, failed or 'checked final desktop refresh')
        if ok then mp.msg.info('PASS: ' .. message)
        else mp.msg.error('FAIL: ' .. message) end
        mp.commandv('quit', ok and 0 or 1)
    end)
end

mp.observe_property('idle-active', 'bool', function(_, idle)
    if idle and loaded >= opts.files then ending = true; finish() end
end)
mp.observe_property('eof-reached', 'bool', function(_, eof)
    if eof and mp.get_property_number('playlist-pos', -1) == opts.files - 1 then
        ending = true
        finish()
    end
end)
mp.add_timeout(45, function()
    mp.msg.error('FAIL: playlist check timed out')
    mp.commandv('quit', 1)
end)
