// 简化的Markdown解析器
function parseMarkdown(text) {
    if (!text) return '';
    
    let html = text;
    
    // 转义HTML特殊字符
    html = html.replace(/&/g, '&amp;')
               .replace(/</g, '&lt;')
               .replace(/>/g, '&gt;');
    
    // 标题
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // 粗体和斜体
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // 代码块
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // 引用
    html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');
    
    // 列表
    html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^(\d+)\. (.*$)/gim, '<li>$1. $2</li>');
    
    // 包装列表项
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    
    // 图片
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; height: auto;">');
    
    // 表格 - 改进处理
    const tableRegex = /(\|.*\|[\s\S]*?)(?=\n\n|\n$|$)/g;
    html = html.replace(tableRegex, function(match) {
        const lines = match.trim().split('\n');
        let tableHtml = '<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">';
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line || !line.includes('|')) continue;
            
            const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell);
            
            // 检查是否是分隔行
            if (cells.every(cell => /^[-:|\s]+$/.test(cell))) {
                continue; // 跳过分隔行
            }
            
            tableHtml += '<tr>';
            cells.forEach(cell => {
                tableHtml += `<td style="border: 1px solid #ddd; padding: 8px;">${cell}</td>`;
            });
            tableHtml += '</tr>';
        }
        
        tableHtml += '</table>';
        return tableHtml;
    });
    
    // 段落处理 - 改进换行处理
    html = html.replace(/\n\n+/g, '</p><p>');
    html = '<p>' + html + '</p>';
    
    // 单行换行处理 - 只在段落内处理
    html = html.replace(/(?<!<\/p>)\n(?!<p>)/g, '<br>');
    
    // 清理空段落和多余换行
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<br>\s*<br>/g, '<br>');
    
    return html;
}

// 兼容marked.parse的接口
window.marked = {
    parse: parseMarkdown
};

