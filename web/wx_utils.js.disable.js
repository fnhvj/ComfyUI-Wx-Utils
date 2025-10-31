// web/wx_utils.js
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.WxUtils",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 为 ExampleNode 节点添加特殊处理
        if (nodeData.name === "ExampleNode") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 如果收到更改 widget 的指令
                if (message?.change_widget) {
                    // 查找 text widget 并更新其值
                    const textWidget = this.widgets?.find(w => w.name === "text");
                    if (textWidget) {
                        textWidget.value = "1111111111111111";
                        // 触发节点更新
                        this.setDirtyCanvas(true);
                    }
                }
                
                // 显示文本信息
                if (message?.string) {
                    // 可以添加状态显示 widget
                    let statusWidget = this.widgets?.find(w => w.name === "status");
                    if (!statusWidget) {
                        statusWidget = ComfyWidgets.STRING(this, "status", ["STRING", { 
                            default: "", 
                            multiline: true 
                        }], app).widget;
                        statusWidget.inputEl.readOnly = true;
                        statusWidget.inputEl.style.fontSize = "8px";
                        statusWidget.inputEl.style.fontFamily = "monospace";
                        statusWidget.inputEl.style.backgroundColor = "#222";
                        statusWidget.inputEl.style.color = "#fff";
                        statusWidget.inputEl.style.padding = "8px";
                    }
                    statusWidget.value = message.string[0];
                }
            };
        }
        
        // 为 WxLoopTextPrompt 节点添加特殊处理
        if (nodeData.name === "WxLoopTextPrompt") {
            // 节点创建时添加状态显示 widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 创建状态显示 widget（在节点创建时就初始化）
                if (!this.statusWidget) {
                    this.statusWidget = ComfyWidgets.STRING(this, "status_info", ["STRING", { 
                        default: "等待执行...",
                        multiline: true 
                    }], app).widget;
                    
                    // 设置样式
                    this.statusWidget.inputEl.readOnly = true;
                    this.statusWidget.inputEl.style.fontSize = "9px";
                    this.statusWidget.inputEl.style.fontFamily = "monospace";
                    this.statusWidget.inputEl.style.backgroundColor = "#222";
                    this.statusWidget.inputEl.style.color = "#fff";
                    this.statusWidget.inputEl.style.padding = "2px";
                    this.statusWidget.inputEl.style.borderRadius = "4px";
                    this.statusWidget.inputEl.style.height = "40px";
                }
                
                return r;
            };
            
            // 处理从后端返回的 UI 信息
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 更新状态信息
                if (message?.text && this.statusWidget) {
                    this.statusWidget.value = message.text[0];
                }
                
                // 从后端获取下一个行号并回填到 next_line widget
                if (message?.next_line) {
                    // 查找 next_line widget 并更新其值
                    const nextLineWidget = this.widgets?.find(w => w.name === "next_line");
                    if (nextLineWidget) {
                        // 使用后端计算的下一个行号
                        nextLineWidget.value = message.next_line[0];
                        // 触发画布更新
                        this.setDirtyCanvas(true);
                    }
                }
                
                // 触发重新绘制
                requestAnimationFrame(() => {
                    this.setSize(this.computeSize());
                });
            };
        }

        // WxLoopModelList 节点处理（原有逻辑正确，无需修改）
        if (nodeData.name === "WxLoopModelList") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message?.text) {
                    let statusWidget = this.widgets?.find(w => w.name === "status");
                    if (!statusWidget) {
                        statusWidget = ComfyWidgets.STRING(this, "status", ["STRING", { 
                            default: "", 
                            multiline: true,
                            rows: 2 
                        }], app).widget;
                        statusWidget.inputEl.readOnly = true;
                        statusWidget.inputEl.style.cssText = `
                            font-size: 8px;
                            font-family: monospace;
                            background: #222;
                            color: #fff;
                            padding: 6px;
                            border: none;
                            border-radius: 2px;
                        `;
                    }
                    statusWidget.value = message.text[0];
                    statusWidget.onChange(); // 原有逻辑已加，正确
                }

                if (message?.next_index) {
                    const nextIndexWidget = this.widgets?.find(w => w.name === "next_index");
                    if (nextIndexWidget) {
                        nextIndexWidget.value = Number(message.next_index[0]);
                        nextIndexWidget.onChange(); // 原有逻辑已加，正确
                        this.setDirtyCanvas(true);
                    }
                }

                if (message?.change_widget) {
                    const textWidget = this.widgets?.find(w => w.name === "text");
                    if (textWidget) {
                        textWidget.onChange();
                        this.setDirtyCanvas(true);
                    }
                }
            };
        }

    }
});