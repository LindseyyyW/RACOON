from utils import compute_cell_boundaries
from refined.inference.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata")

def get_column_wise_spans(context, table, table_cols):
    """
    Perform entity linking on table data using different context types.
    
    Args:
        context: The context type for entity linking, one of:
                - 'cell': Each cell processed independently
                - 'col': Entire column processed together
                - 'table': Entire table processed together (column-wise)
                - 'hybrid': Row + column hybrid context
        table: Row-wise representation of the table (list of rows)
        table_cols: Column-wise representation of the table (list of columns)
    
    Returns:
        column_wise_table_spans: A list where each element corresponds to a column,
                                and each column contains a list of entity spans for
                                each cell in that column.
    """
    column_wise_table_spans = []
    
    if context == 'cell':
        # Process each cell independently
        for col in table_cols:
            col_spans = []
            for cell in col:
                text = "|| " + (cell if cell.strip() else "-") + " ||"
                cell_boundaries = compute_cell_boundaries(text)
                spans, cell_sep_spans = refined.process_text(text=text, cell_boundaries=cell_boundaries)
                if len(cell_sep_spans) > 0:
                    col_spans.append(cell_sep_spans[0])
                else:
                    col_spans.append([])
            column_wise_table_spans.append(col_spans)
    
    elif context == 'col':
        # Process each column as a whole
        for col in table_cols:
            text = "|| " + " || ".join(item if item.strip() else "-" for item in col) + " ||"
            cell_boundaries = compute_cell_boundaries(text)
            spans, cell_sep_spans = refined.process_text(text=text, cell_boundaries=cell_boundaries)
            if len(cell_sep_spans) == 0:
                cell_sep_spans = [[] for _ in range(len(col))]
            column_wise_table_spans.append(cell_sep_spans)
    
    elif context == 'table':
        # Process entire table at once (column-wise ordering)
        table_text = "|| " + " || ".join(
            " || ".join(item if item.strip() else "-" for item in col) for col in table_cols
        ) + " ||"
        
        cell_boundaries = compute_cell_boundaries(table_text)
        spans, cell_sep_spans = refined.process_text(text=table_text, cell_boundaries=cell_boundaries)
        
        for col_idx in range(len(table_cols)):
            col_spans = []
            length = len(table_cols[col_idx])
            for row_idx in range(length):
                flat_idx = col_idx * length + row_idx
                if flat_idx < len(cell_sep_spans):
                    col_spans.append(cell_sep_spans[flat_idx])
                else:
                    col_spans.append([])
            column_wise_table_spans.append(col_spans)
    
    elif context == 'hybrid':
        # Process with row + column hybrid context
        table_spans = []
        for n, row in enumerate(table):
            row_spans = []
            for cid, cell in enumerate(row):
                linked_en = []
                text = "|| " + " || ".join(item if item.strip() else "-" for item in row) + \
                       " || " + " || ".join(table[r][cid] if table[r][cid].strip() else "-" for r in range(len(table)))
                cell_boundaries = compute_cell_boundaries(text)
                spans, cell_sep_spans = refined.process_text(text=text, cell_boundaries=cell_boundaries)
                if len(cell_sep_spans) != 0:
                    linked_en = cell_sep_spans[cid]
                row_spans.append(linked_en)
            table_spans.append(row_spans)
        column_wise_table_spans = list(zip(*table_spans))
    
    return column_wise_table_spans
