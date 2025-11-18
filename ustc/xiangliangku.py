# init_and_upload.py
from chunks2embedding import embedding_init, upsert_md_file,get_collection_info,delete_by_source
import os


def main():
    # 配置
    collection_name = "xinxi"
    md_dir = "/home/easyai/ragmd"  # 您的Markdown文件目录

    # 初始化向量存储
    print("�� 初始化向量存储...")
    vector_store = embedding_init(host="localhost", port=6333, collection_name=collection_name)

    # 上传所有Markdown文件
    print("\n�� 开始上传知识库...")
    success_count = 0
    fail_count = 0

    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            try:
                file_path = os.path.join(md_dir, filename)
                print(f"\n�� 正在处理: {filename}")
                upsert_md_file(file_path, vector_store)
                success_count += 1
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {str(e)}")
                fail_count += 1

    # 总结
    print("\n✅ 上传完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")

    # 显示集合信息

    get_collection_info(collection_name, host="localhost", port=6333)


if __name__ == "__main__":
    main()