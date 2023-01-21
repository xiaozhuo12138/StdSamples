
local igwin = require"imgui.window"
--local win = igwin:SDL(800,400, "dial")
local win = igwin:GLFW(800,400, "dial")

local gui = require"filebrowser"(win.ig)

local fb = gui.FileBrowser(nil,{key="loader",pattern="%.lua"},function(fname) print("load",fname) end)
local fbs = gui.FileBrowser(nil,{key="saver",check_existence=true},function(fname) print("save",fname) end)

function win:draw(ig)
	if ig.SmallButton("load") then
        fb.open()
    end
    fb.draw()
    
    if ig.SmallButton("save") then
        fbs.open()
    end
    fbs.draw()
end

win:start()