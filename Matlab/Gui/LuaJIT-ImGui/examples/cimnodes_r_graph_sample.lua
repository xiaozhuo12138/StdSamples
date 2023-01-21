local igwin = require"imgui.window"

--local win = igwin:SDL(800,400, "compute graph",{vsync=true})
local win = igwin:GLFW(800,400, "compute graph",{vsync=false})
local ig = win.ig
local ffi = require"ffi"
local serializer = require"serializer"

local function DFS(G,v)
    G.nodes_explored[v] = true
    local values = {}
    for i,id in ipairs(G.nodes[v].fromedges) do
        local value
        if not G.nodes_explored[id] then
            value = DFS(G,id)
            G.nodes_values[id] = value
        else
            value = G.nodes_values[id]
            if not value then 
                value = G.old_nodes_values[id]
                --print("cicle found",value)
            end
        end
        table.insert(values,value)
    end
    return G.nodes[v].compute(values)
end

local function Graph()
    local G = {nodes={},edges={}}
    function G:insert_nodeIn(id, compute)
        self.nodes[id] = {fromedges={},compute=compute,kind="in"}
    end
    function G:insert_nodeOut(id, compute)
        self.nodes[id] = {fromedges={},compute=compute,kind="out"}
    end
    function G:insert_node(id, compute)
        self.nodes[id] = {fromedges={},compute=compute,kind="?"}
    end
    function G:delete_node(id)
        local fromedges = self.nodes[id].fromedges
        for i,id1 in ipairs(fromedges) do
            self:delete_edge(id1,id)
        end
        self.nodes[id] = nil
    end
    function G:insert_edge(id1,id2)
        local fromedgs = self.nodes[id2].fromedges
        table.insert(fromedgs,id1)
    end
    function G:delete_edge(id1,id2)
        local fromedgs = self.nodes[id2].fromedges
        for i,id in ipairs(fromedgs) do
            if id == id1 then table.remove(fromedgs,i) end
        end
    end
    function G:DFS_prepare()
        G.nodes_explored = {}
        G.old_nodes_values = G.nodes_values or {}
        G.nodes_values = {}
    end
    function G:DFS(root)
        return DFS(self,root)
    end
    return G
end

local function Connection()
    local link = {
        input_node=ffi.new("void*[1]"),
        input_slot=ffi.new("const char*[1]"),
        output_node=ffi.new("void*[1]"),
        output_slot=ffi.new("const char*[1]")
    }
    return link
end
local function idtokey(v)
    return tonumber(ffi.cast("uintptr_t",v))
end
local function pinkey(isin,node,slotname)
    return (isin and "in" or "out")..idtokey(node.id)..slotname
end
local function Node(value,editor,typen,loadT)
    local node
    if not loadT then
        node = {
            id = ffi.cast("void*",editor:newid()),
            pos = ig.ImVec2(20,20),
            selected = ffi.new"bool[1]",
            title = typen.name,
            is_root = typen.is_root
        }
        node.nins = #typen.ins
        node.nouts = #typen.outs
        node.input_slots = ffi.new("SlotInfo[?]",node.nins,typen.ins)
        node.output_slots = ffi.new("SlotInfo[?]",node.nouts,typen.outs)
        node.connections = {}
        --create input_id
        node.inputs = {}
        for i=1,node.nins do
            node.inputs[i] = pinkey(true,node,typen.ins[i][1])
        end
        --create output_id
        node.outputs = {}
        for i=1,node.nouts do
            node.outputs[i] = pinkey(false,node,typen.outs[i][1])
        end
        -- create static_id
        node.values = {}
        for i=1,node.nins do
            node.values[i] = ffi.new("float[?]",1,value)
        end
    else
        node = loadT
    end
    local typename = node.title
    for i,typ in ipairs(editor.nodetypes) do
        if typename==typ.name then
            node.compute = typ.compute
            node.show = typ.show
            break
        end
    end
    -------------add pins to Graph
    local function computeIn(i)
        return function(t)
            if t[1] then
                return t[1]
            else
                return node.values[i][0]
            end
        end
    end
    
    for i ,input_id in ipairs(node.inputs) do
        editor.G:insert_node(input_id,computeIn(i))
    end
    if node.nouts > 0 then 
        for i,output_id in ipairs(node.outputs) do
            editor.G:insert_node(output_id,node.compute)
            for _ ,input_id in ipairs(node.inputs) do
                editor.G:insert_edge(input_id,output_id)
            end
        end
    end
    --add root node
    if node.is_root then
        editor.G:insert_node(idtokey(node.id),node.compute)
        for _ ,input_id in ipairs(node.inputs) do
            editor.G:insert_edge(input_id,idtokey(node.id))
        end
    end
    function node:delete()
        --delete pins from graph
        for _ ,input_id in ipairs(self.inputs) do
            editor.G:delete_node(input_id)
        end
        for _ ,output_id in ipairs(self.outputs) do
            editor.G:delete_node(output_id)
        end
        if node.is_root then
            editor.G:delete_node(idtokey(node.id))
        end
        editor.nodes[idtokey(self.id)] = nil
    end
    function node:delete_connection(conn)
        for i,c in ipairs(self.connections) do
            if c.input_node[0]==conn.input_node[0] and
            c.input_slot[0]==conn.input_slot[0] and
            c.output_node[0]==conn.output_node[0] and
            c.output_slot[0]==conn.output_slot[0] then
                table.remove(self.connections,i)
                return
            end
        end
    end
    function node:save_str(name)
        return serializer(name,self)
    end
    function node:draw()

        if ig.ImNodes_Ez_BeginNode(node.id,node.title,node.pos,node.selected) then

            ig.ImNodes_Ez_InputSlots(node.input_slots, node.nins);
            for i, input_id in ipairs(node.inputs) do
                --if there is no input
                local orig = editor.G.nodes[input_id].fromedges
                    ig.PushItemWidth(80.0);
                    if #orig==0 then
                        ig.DragFloat("##value"..i, node.values[i], 0.01);
                    else
                        ig.Dummy(ig.ImVec2(80,ig.GetTextLineHeightWithSpacing()))
                    end
                    ig.PopItemWidth();
                
            end
            ig.ImNodes_Ez_OutputSlots(node.output_slots, node.nouts);
            
            local conn = Connection()
            if (ig.ImNodes_GetNewConnection(conn.input_node, conn.input_slot, conn.output_node, conn.output_slot)) then
                --only one link for input
                local iid = pinkey(true,editor.nodes[idtokey(conn.input_node[0])],ffi.string(conn.input_slot[0]))
                local oid = pinkey(false,editor.nodes[idtokey(conn.output_node[0])],ffi.string(conn.output_slot[0]))
                local dest = editor.G.nodes[iid].fromedges
                if #dest==0 then
                    editor.G:insert_edge(oid,iid)
                    table.insert(editor.nodes[idtokey(conn.input_node[0])].connections,conn)
                    table.insert(editor.nodes[idtokey(conn.output_node[0])].connections,conn)
                end
            end
            
            for i,conn in ipairs(node.connections) do
                if conn.input_node[0] == node.id then
                    if not ig.ImNodes_Connection(conn.input_node[0], conn.input_slot[0], conn.output_node[0], conn.output_slot[0]) then
                        local iid = pinkey(true,editor.nodes[idtokey(conn.input_node[0])],ffi.string(conn.input_slot[0]))
                        local oid = pinkey(false,editor.nodes[idtokey(conn.output_node[0])],ffi.string(conn.output_slot[0]))
                        editor.G:delete_edge(oid,iid)
                        editor.nodes[idtokey(conn.input_node[0])]:delete_connection(conn)
                        editor.nodes[idtokey(conn.output_node[0])]:delete_connection(conn)
                    end
                end
            end
        end
        ig.ImNodes_Ez_EndNode();
        
        local dodelete = false
        local user_key = ig.GetKeyIndex(ig.lib.ImGuiKey_X)
        if ig.IsWindowFocused(ig.lib.ImGuiFocusedFlags_RootAndChildWindows) and ig.IsKeyReleased(user_key)
        then
            dodelete = true
        end
        if node.selected[0] and dodelete then
            for i,conn in ipairs(node.connections) do
                if conn.output_node[0] == node.id then
                    local iid = pinkey(true,editor.nodes[idtokey(conn.input_node[0])],ffi.string(conn.input_slot[0]))
                    local oid = pinkey(false,editor.nodes[idtokey(conn.output_node[0])],ffi.string(conn.output_slot[0]))
                    editor.G:delete_edge(oid,iid)
                    editor.nodes[idtokey(conn.input_node[0])]:delete_connection(conn)
                else
                    local iid = pinkey(true,node,ffi.string(conn.input_slot[0]))
                    local oid = pinkey(false,editor.nodes[idtokey(conn.output_node[0])],ffi.string(conn.output_slot[0]))
                    editor.G:delete_edge(oid,iid)
                    editor.nodes[idtokey(conn.output_node[0])]:delete_connection(conn)
                end
            end
            node.connections = {}
            if node.is_root then
                for i,v in ipairs(editor.root_nodes) do
                    if v == idtokey(node.id) then
                        table.remove(editor.root_nodes,i)
                    end
                end
            end
            node:delete()
        end
    end
    return node
end

local function show_editor(editor)
    ig.Begin(editor.name);

    ig.TextUnformatted("A -- add node");
    ig.TextUnformatted("X -- delete selected node or link");

    ig.ImNodes_BeginCanvas(editor.context);

    for _, node in pairs(editor.nodes) do
        node:draw()
    end
    
    local user_key = ig.GetKeyIndex(ig.lib.ImGuiKey_A)
    if (ig.IsWindowFocused(ig.lib.ImGuiFocusedFlags_RootAndChildWindows) and ig.IsKeyReleased(user_key))
    then
        ig.OpenPopup("add node")
    end
    local window_pos = ig.GetWindowPos()
    if ig.BeginPopup"add node" then
        local click_pos = ig.GetMousePosOnOpeningCurrentPopup();
        for i,ntype in ipairs(editor.nodetypes) do
            if ig.MenuItem(ntype.name) then
                local newnode = editor:Node(0,ntype)
                newnode.pos = click_pos - window_pos 
            end
        end
        ig.EndPopup()
    end
    ig.ImNodes_EndCanvas()
    ig.End();
    
        -- The outputs
    local outs =  editor:evaluate()
    for i,root in ipairs(editor.root_nodes) do
        editor.nodes[root]:show(outs[i],i)
    end
end
local function Editor(name, nodetypes)
    local E = {nodes={},current_id=0,name=name,nodetypes = nodetypes,root_nodes={}}
    E.G = Graph()
    function E:evaluate()
        local outs = {}
        self.G:DFS_prepare()
        for i,root in ipairs(self.root_nodes) do
            outs[i] = self.G:DFS(root)
        end
        return outs
    end
    function E:newid()
        E.current_id = E.current_id + 1
        return E.current_id
    end
    function E:Node(value, typen)
        local newnode = Node(value,self, typen)
        self.nodes[idtokey(newnode.id)] = newnode
        if typen.is_root then table.insert(self.root_nodes, idtokey(newnode.id)) end
        return newnode
    end
    
    E.draw = show_editor

    function E:save_str()
        local str = [[local ffi = require"ffi"]]
        str = str .. "\n"
        for k,node in pairs(self.nodes) do
            str = str .. node:save_str("node"..k) .. "\n"
        end
        str = str .. serializer("root_nodes",self.root_nodes)
        str = str .. "return {nodes = {"
        for k,node in pairs(self.nodes) do
            local kst = type(k)=="number" and "["..k.."]" or k
            str = str .. kst .. "=" .. ("node"..k) .. ","
        end
        str = str .. "},name='"..self.name
        str = str .. "',current_id = " .. self.current_id 
        str = str .. ",root_nodes = root_nodes}"
        return str
    end
    function E:save()
        local str = self:save_str()
        local file,err = io.open(self.name.."_saved","w")
        if not file then print(err);error"opening file" end
        file:write(str)
        file:close()
    end
    function E:load()
        local file,err = io.open(self.name.."_saved","r")
        if file then
            local str = file:read"*a"
            file:close()
            self:load_str(str)
        end
    end
    function E:load_str(str)
        self.nodes = {}
        local f = loadstring(str)
        setfenv(f,setmetatable({ig=ig},{ __index = _G}))
        local loadedE = f()
        for k,v in pairs(loadedE.nodes) do
            local node = Node(0,self,nil,v)
            self.nodes[idtokey(node.id)] = node
        end
        --connections
        for k,node in pairs(self.nodes) do
            for i,conn in ipairs(node.connections) do
                if conn.input_node[0] == node.id then
                    local iid = pinkey(true,self.nodes[idtokey(conn.input_node[0])],ffi.string(conn.input_slot[0]))
                    local oid = pinkey(false,self.nodes[idtokey(conn.output_node[0])],ffi.string(conn.output_slot[0]))
                    self.G:insert_edge(oid,iid)
                end
            end
        end
        self.current_id = loadedE.current_id
        --self.name = loadedE.name
        self.root_nodes = loadedE.root_nodes
    end
    E.context = ig.CanvasState();
    return E
end
---------------------------------------use it!!-------------------------------------
local function clamp(v)
    return math.max(0,math.min(1,v))
end

local nodetypes = {
{   name = "add",
    ins = {{"lhs",1},{"rhs",1}},
    outs = {{"sum",1}},
    compute = function(t)
        return t[1] + t[2]
    end
},{
    name = "multiply",
    ins = {{"lhs",1},{"rhs",1}},
    outs = {{"mul",1}},
    compute = function(t)
        return t[1] * t[2]
    end
},{
    name = "output",
    ins = {{"r",1},{"g",1},{"b",1}},
    is_root = true,
    outs = {},
    show = function(self,v,i)
        ig.PushStyleColor(ig.lib.ImGuiCol_WindowBg, v);
        ig.Begin("output color"..i);
        ig.End();
        ig.PopStyleColor();
    end,
    compute = function(t)
        local a,b,c = clamp(t[1]),clamp(t[2]),clamp(t[3])
        return ig.U32(a,b,c)
    end
},{
    name = "lisa",
    ins = {{"x",1},{"y",1}},
    outs = {},
    is_root = true,
    show = function(self,v,i)
        local lisaS = 30
        self.lisamem = self.lisamem or {}
        local lisamem = self.lisamem
        ig.Begin("output lisa"..i);
        ig.Text("x: %f, y: %f",v[1],v[2])
        local canvas_p0 = ig.GetCursorScreenPos();  
        local canvas_sz = ig.GetContentRegionAvail();
        local canvas_p1 = canvas_p0 + canvas_sz
        local draw_list = ig.GetWindowDrawList();
        draw_list:AddRectFilled(canvas_p0, canvas_p1, ig.U32(50/255, 50/255, 50/255, 1));
        draw_list:AddRect(canvas_p0, canvas_p1, ig.U32(1, 1, 1, 1));
        table.insert(lisamem ,1,v)
        table.remove(lisamem,lisaS+1)
        for i=1,lisaS do
            local u = lisamem[i] or v
            draw_list:AddCircleFilled(ig.ImVec2(u[1]*canvas_sz.x,u[2]*canvas_sz.y)+canvas_p0, 3, ig.U32(1,1,1,1));
        end
        ig.End();
    end,
    compute = function(t)
        local x,y = clamp(t[1]),clamp(t[2])
        return {x,y}
    end
},{
    name = "sine",
    ins = {{"in",1}},
    outs = {{"out",1}},
    compute = function(t)
        return math.sin(t[1])*0.5 + 0.5
    end
},{
    name = "time",
    ins = {},
    outs = {{"time",1}},
    compute = function(t)
        return os.clock()
    end
}
}

local editor1 = Editor("compute_graph_r",nodetypes)
editor1:load()

function win:draw(ig)
    editor1:draw()
    ig.ShowDemoWindow()
end

local function clean()
  editor1:save()
end

win:start(clean)