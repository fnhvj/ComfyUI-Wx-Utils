// web/wx_utils.js
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.WxUtils",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 为 WxLoopTextPrompt 节点添加特殊处理
        if (nodeData.name === "WxLoopTextPrompt") {
            // 创建用于显示状态信息的 widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 创建状态显示 widget
                this.statusWidget = ComfyWidgets.STRING(this, "status", ["STRING", { multiline: true }], app).widget;
                this.statusWidget.inputEl.readOnly = true;
                this.statusWidget.inputEl.style.fontSize = "12px";
                this.statusWidget.inputEl.style.fontFamily = "monospace";
                this.statusWidget.inputEl.style.backgroundColor = "#222";
                this.statusWidget.inputEl.style.color = "#fff";
                this.statusWidget.inputEl.style.height = "100px";
                
                return r;
            };
            
            // 处理从后端返回的 UI 信息
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 更新状态信息
                if (message?.text?.length) {
                    this.statusWidget.value = message.text[0];
                    this.statusWidget.inputEl.title = message.text[0]; // 鼠标悬停时显示完整信息
                }
                
                // 触发重新绘制
                requestAnimationFrame(() => {
                    this.setSize(this.computeSize());
                });
            };
        }
    }
});