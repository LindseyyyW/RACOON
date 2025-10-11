from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def pruning_rel(table, hint):
    """
    Use an LLM to prune the retrieved entity relations from the KG. 
    Returns: pruned hint.
    """
    messages = [
        SystemMessage(content="""
        You are a careful and precise assistant. Your task is to prune a hint block associated with tabular data.
        ONLY remove parts of the original hint that are clearly incorrect, irrelevant, redundant, or do not align with the actual column values.

        DO NOT add new interpretations, rephrase, or insert descriptions that were not in the original hint.
        DO NOT change the structure or wording of any valid content.
        DO NOT add new types or labels.

        Keep the output format exactly the same:
        - Column 1 & Column N: <original hint with only clearly unhelpful or inaccurate parts removed>
        
        Include each column pair in your output even if the pruned hint is empty for that column pair. Your output should be a string, not a list. 
                      
        If a column's hint is already empty, keep it that way.
        Return only the pruned hint block.
                """),
                HumanMessage(content=f"""
        Table:
        {table}

        Original hint:
        {hint}
                """)
    ]
    response = llm(messages)
    return response.content.strip()

def pruning_orig(table, hint):
    """
    Use an LLM to prune the retrieved hint (entity labels, entity labels + descriptions, or entity types) from the KG.
    Returns: pruned hint.
    """
    messages = [
        SystemMessage(content="""
        You are a careful and precise assistant. Your task is to prune a hint block associated with tabular data.
        ONLY remove parts of the original hint that are clearly incorrect, irrelevant, redundant, or do not align with the actual column values.

        DO NOT add new interpretations, rephrase, or insert descriptions that were not in the original hint.
        DO NOT change the structure or wording of any valid content.
        DO NOT add new types or labels.

        Keep the output format exactly the same:
        - Column N: <original hint with only clearly unhelpful or inaccurate parts removed>
        
        Your output should be a string, not a list. Your output should include all the columns in the original hint even if the pruned hint is empty for some columns.
                      
        If a column's hint is already empty, keep it that way.
        Return only the pruned hint block.
                """),
                HumanMessage(content=f"""
        Table:
        {table}

        Original hint:
        {hint}
                """)
    ]
    response = llm(messages)
    return response.content.strip()


