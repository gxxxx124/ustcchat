
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from operator import add
import re
from langgraph.checkpoint.postgres import PostgresSaver
import os

# 1. 配置模型
inference_server_url = "http://localhost:11434/v1"
model = ChatOpenAI(
    model="qwen3:4b",
    openai_api_key="none",
    openai_api_base=inference_server_url,
    max_tokens=500,
    temperature=0.7,  # 降低温度提高决策稳定性
)

# 2. 创建工具实例
from rag_tool import create_rag_tool
from searxng_server import create_search_tool

rag_tool = create_rag_tool(
    host="localhost",
    port=6333,
    collection_name="test"
)

web_search_tool = create_search_tool()


# 3. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]  # 只保留消息历史


# 4. 创建模型调用节点（保持不变）
def call_model(state: AgentState):
    """模型自主决策是否需要调用工具"""
    messages = state["messages"]

    # 如果是初始查询，添加系统提示
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        system_prompt = """你是一个智能助手，可以使用以下工具：

        1. rag_knowledge_search: 查询内部知识库（主要包含医学相关内容）
        2. web_search: 查询最新互联网信息

        工作流程：
        - 首先尝试使用 rag_knowledge_search
        - 如果结果不相关、过时或信息不足，再使用 web_search
        - 确保最终回答整合所有可用信息

        特别注意：
        - 内部知识库主要包含医学领域内容
        - 对于科技、工程、最新进展类问题，RAG结果通常不完整
        - 当RAG返回医学相关内容但问题涉及其他领域时，必须调用web_search
        """
        messages = [SystemMessage(content=system_prompt)] + messages

    # 始终绑定所有可用工具
    web_search_count = sum(1 for m in messages
                           if isinstance(m, AIMessage) and
                           m.tool_calls and
                           m.tool_calls[0]["name"] == "web_search_tool")

    # 如果已经搜索3次以上，强制生成最终回答
    if web_search_count >= 3:
        model_with_tools = model  # 移除工具绑定
    else:
        model_with_tools = model.bind_tools([rag_tool, web_search_tool])

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# 5. 创建条件边函数（保持不变）
def should_continue(state: AgentState):
    """检查是否需要调用工具"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果模型要求调用工具，则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则结束
    return END


# 7. 正确实现短期对话记忆
if __name__ == "__main__":
    # 从环境变量获取数据库连接字符串（更安全的方式）
    DB_URI = os.getenv(
        "DB_URI",
        "postgresql://postgres:postgres@localhost:5432/langgraph_db?sslmode=disable"
    )

    # >>>> 关键修改：在with块内创建和使用graph <<<<
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # 首次运行时需要设置表结构（取消注释一次即可）
        # checkpointer.setup()

        # 构建图
        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode([rag_tool, web_search_tool]))
        builder.add_edge(START, "call_model")

        builder.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        builder.add_edge("tools", "call_model")

        # 编译图
        graph = builder.compile(checkpointer=checkpointer)

        # 为每个会话定义唯一的thread_id
        config = {"configurable": {"thread_id": "user_session_12"}}

        # 第一轮对话
        print("=" * 50)
        print("第一轮对话")
        initial_state = {
            "messages": [HumanMessage(content="NGS适用于哪些人群")]
        }

        # 打印执行过程
        for step in graph.stream(initial_state, config=config):
            print("\n" + "=" * 50)
            print(f"状态更新: {list(step.keys())[0]}")
            last_msg = step[list(step.keys())[0]]["messages"][0]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"��️ 模型决定调用工具: {last_msg.tool_calls[0]['name']}")
                print(f"�� 查询参数: {last_msg.tool_calls[0]['args']['query']}")
            else:
                print(f"�� 模型响应: {last_msg.content}")

        # 第二轮对话
        print("\n" + "=" * 50)
        print("第二轮对话")
        second_query = {
            "messages": [HumanMessage(content="for 2024年巴黎奥运会乒乓球男单冠军是谁")]
        }

        for step in graph.stream(second_query, config=config):
            print("\n" + "=" * 50)
            print(f"状态更新: {list(step.keys())[0]}")
            last_msg = step[list(step.keys())[0]]["messages"][0]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"��️ 模型决定调用工具: {last_msg.tool_calls[0]['name']}")
                print(f"�� 查询参数: {last_msg.tool_calls[0]['args']['query']}")
            else:
                print(f"�� 模型响应: {last_msg.content}")

        # 查看完整对话历史（可选）
        final_state = graph.get_state(config)
        print("\n" + "=" * 50)
        print("完整对话历史:")
        for i, msg in enumerate(final_state.values["messages"]):
            if isinstance(msg, HumanMessage):
                print(f"[{i}] �� 用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"[{i}] �� 助手: {msg.content}")
            elif isinstance(msg, SystemMessage):
                print(f"[{i}] ⚙️ 系统提示")