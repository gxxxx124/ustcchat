import re
import os
import json
from typing import List, Dict, Any


def parse_markdown_file(file_path, max_chunk_size=1000, overlap_size=200):
    """
    改进版Markdown解析器，增强标题识别和内容分割能力
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 预处理：合并多余空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 增强标题识别：排除非标题的#字符行
    heading_pattern = re.compile(
        r'^(#{1,6})\s+([^\n#]+?)(?:\n|$)',  # 确保标题不含#字符
        re.MULTILINE
    )

    # 提取所有标题及其位置
    headings = []
    for match in heading_pattern.finditer(content):
        start_pos = match.start()
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((start_pos, level, title))

    # 添加文档结束标记
    headings.append((len(content), 0, "EOF"))

    # 处理文档头部非标题内容
    chunks = []
    if headings and headings[0][0] > 0:
        header_content = content[:headings[0][0]].strip()
        if header_content:
            chunks.append({
                'title': '文档头部',
                'title_text': '文档头部',
                'content_text': header_content,
                'level': 0,
                'parent_title': None,
                'path': '文档头部',
                'source': os.path.basename(file_path)
            })

    # 维护标题层级栈
    title_stack = []

    # 遍历所有标题区块
    for i in range(len(headings) - 1):
        start_pos, level, title = headings[i]
        next_start_pos, _, _ = headings[i + 1]

        # 计算标题行结束位置
        title_end_pos = content.find('\n', start_pos)
        if title_end_pos == -1:
            title_end_pos = start_pos + len(title) + level + 1

        # 提取当前标题下的内容（跳过标题行）
        section_content = content[title_end_pos:next_start_pos].strip()

        # 更新标题层级栈
        while title_stack and title_stack[-1]['level'] >= level:
            title_stack.pop()

        # 构建标题路径
        parent_title = title_stack[-1]['title'] if title_stack else None
        path_parts = [t['title'] for t in title_stack]
        path_parts.append(title)
        path = ' / '.join(path_parts)

        # 将当前标题压入栈
        title_stack.append({'title': title, 'level': level})

        # 空内容处理
        if not section_content:
            continue

        # 处理大块内容分割
        if len(section_content) > max_chunk_size:
            sub_chunks = split_large_content_with_metadata(
                section_content,
                max_chunk_size,
                overlap_size,
                title=title,
                level=level,
                parent_title=parent_title,
                path=path,
                source=os.path.basename(file_path))
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                'title': title,
                'title_text': title,
                'content_text': section_content,
                'level': level,
                'parent_title': parent_title,
                'path': path,
                'source': os.path.basename(file_path)
            })

            # 添加上下文重叠（仅限同级标题）
            chunks = add_smart_overlap(chunks, overlap_size)
            
    return chunks


def parse_markdown_file_api(file_path: str, overlap_size: int = 200) -> List[Dict[str, Any]]:
    """解析Markdown文件的API版本，不保存JSON文件"""
    chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    current_chunk = ""
    current_title = ""
    current_level = 0
    parent_title = ""
    
    for line in lines:
        line = line.strip()
        
        # 检查是否是标题
        if line.startswith('#'):
            # 保存当前chunk
            if current_chunk.strip():
                chunks.append({
                    'content_text': current_chunk.strip(),
                    'title_text': current_title,
                    'level': current_level,
                    'parent_title': parent_title,
                    'path': f"{current_title}",
                    'source': os.path.basename(file_path)
                })
            
            # 开始新的chunk
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            
            # 更新parent_title
            if level > current_level:
                parent_title = current_title
            elif level < current_level:
                parent_title = ""
            
            current_chunk = line + '\n'
            current_title = title
            current_level = level
            
        else:
            current_chunk += line + '\n'
    
    # 保存最后一个chunk
    if current_chunk.strip():
        chunks.append({
            'content_text': current_chunk.strip(),
            'title_text': current_title,
            'level': current_level,
            'parent_title': parent_title,
            'path': f"{current_title}",
            'source': os.path.basename(file_path)
        })
    
    # 处理大内容分块
    final_chunks = []
    for chunk in chunks:
        if len(chunk['content_text']) > 1000:  # 如果内容太长，进行分块
            sub_chunks = split_large_content_with_metadata(
                chunk['content_text'], 1000, overlap_size,
                chunk['title_text'], chunk['level'], chunk['parent_title'],
                chunk['path'], chunk['source']
            )
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    # 添加上下文重叠（仅限同级标题）
    final_chunks = add_smart_overlap(final_chunks, overlap_size)
    
    return final_chunks


def split_large_content_with_metadata(content, max_size, overlap_size,
                                      title, level, parent_title, path, source):
    """
    改进的大内容块分割，保持段落完整
    """
    # 增强段落分割：处理各种换行情况
    paragraphs = []
    current_para = ""

    for line in content.split('\n'):
        stripped_line = line.strip()
        if not stripped_line:  # 空行表示段落结束
            if current_para:
                paragraphs.append(current_para.strip())
                current_para = ""
        else:
            # 处理列表项、引用等特殊结构
            if re.match(r'^[\-\*•>]\s+', stripped_line) and current_para:
                paragraphs.append(current_para.strip())
                current_para = stripped_line + " "
            else:
                current_para += stripped_line + " "

    if current_para:
        paragraphs.append(current_para.strip())

    chunks = []
    current_chunk = ""
    chunk_index = 1

    for para in paragraphs:
        # 处理超大段落（超过最大块尺寸的80%）
        if len(para) > max_size * 0.8:
            # 按句子分割超大段落
            sentences = re.split(r'(?<=[.!?。！？])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 > max_size:
                    if current_chunk:
                        chunks.append(create_chunk(
                            title, chunk_index, current_chunk.strip(),
                            level, parent_title, path, source))
                        current_chunk = ""
                        chunk_index += 1
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
        else:
            if len(current_chunk) + len(para) + 2 > max_size:
                if current_chunk:
                    chunks.append(create_chunk(
                        title, chunk_index, current_chunk.strip(),
                        level, parent_title, path, source))
                    current_chunk = ""
                    chunk_index += 1
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"

    if current_chunk:
        chunks.append(create_chunk(
            title, chunk_index, current_chunk.strip(),
            level, parent_title, path, source))

    return chunks


def create_chunk(title, index, content, level, parent_title, path, source):
    """创建标准化的块结构"""
    chunk_title = f"{title}" if index == 1 else f"{title} (Part {index})"
    return {
        'title': chunk_title,
        'title_text': title,
        'content_text': content,
        'level': level,
        'parent_title': parent_title,
        'path': path,
        'source': source
    }


def add_smart_overlap(chunks, overlap_size=100):
    """
    智能添加上下文重叠，仅在同级同路径块间添加
    """
    if not chunks:
        return []

    # 添加第一个块（无重叠）
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        current = chunks[i]
        prev = chunks[i - 1]

        # 仅当属于同一章节时添加重叠
        if (current['path'] == prev['path'] and
                current['level'] == prev['level']):
            # 提取前一块的尾部内容
            prev_content = prev['content_text']
            overlap_text = prev_content[-overlap_size:] if len(prev_content) > overlap_size else prev_content

            # 添加重叠内容
            current['content_text'] = overlap_text + '\n\n' + current['content_text']

        result.append(current)

    return result


def save_chunks_to_json(chunks, output_path):
    """保存处理结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)