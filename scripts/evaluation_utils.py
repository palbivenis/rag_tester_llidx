from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from llama_index.core.evaluation import EvaluationResult

from collections import defaultdict
from typing import List
from concurrent import futures
from tqdm import tqdm


def get_answers_source_nodes(responses)->Tuple[list, list]:

    answers = []
    sources = []
        
    for response in responses:
        answers.append(response.response)
        text_md = ""
        
        for n in response.source_nodes:
            
            text_md += (
                f"**Node ID:** {n.node.node_id}{chr(10)}"
                f"**Similarity:** {n.score}{chr(10)}"
                f"**Text:** {n.node.get_content()}{chr(10)}"
                f"**Metadata:** {n.node.metadata}{chr(10)}"
                f"~~~~{chr(10)}"
            )
        sources.append(text_md)
    
    return answers, sources

def get_summary_scores_df(
    eval_results_list: List[EvaluationResult], names: List[str], metric_keys: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get results df.

    Args:
        eval_results_list (List[EvaluationResult]):
            List of evaluation results.
        names (List[str]):
            Names of the evaluation results.
        metric_keys (List[str]):
            List of metric keys to get.

    """
    mean_dict = defaultdict(list)
    mean_dict["score"] = "Mean Score"
    
    sum_dict = defaultdict(list)
    sum_dict["score"] = "Total Score"

    for metric_key in metric_keys:
        for eval_results in eval_results_list:
            mean_score = np.array([r.score for r in eval_results[metric_key]]).mean()
            sum_score = np.array([r.score for r in eval_results[metric_key]]).sum()
            mean_dict[metric_key].append(mean_score)
            sum_dict[metric_key].append(sum_score)
    
    return pd.DataFrame(mean_dict), pd.DataFrame(sum_dict)


def get_eval_results_df(
    query_nums: List[str], expected_answers: List[str], results_arr: List[EvaluationResult], metric: Optional[str] = None
) -> pd.DataFrame:
    """Organizes EvaluationResults into a deep dataframe and computes the mean
    score.

    result:
        result_df: pd.DataFrame representing all the evaluation results
        
    """
    if len(query_nums) != len(results_arr):
        raise ValueError("names and results_arr must have same length.")

    qs = []
    ss = []
    fs = []
    rs = []
    cs = []
    for res in results_arr:
        qs.append(res.query)
        ss.append(res.score)
        fs.append(res.feedback)
        rs.append(res.response)
        cs.append(res.contexts)

    deep_df = pd.DataFrame(
        {
            "query_num": query_nums,
            "query": qs,
            "expected_answer": expected_answers,
            "generated_answer": rs,
            "score": ss,
            "feedback": fs,
            "contexts": cs
        }
    )
    
    return deep_df


def threadpool_map(f, args_list: List[dict], num_workers: int = 64, return_exceptions: bool = True) -> list:
    """
    Same as ThreadPoolExecutor.map with option of returning exceptions. Returns results in the same
    order as input `args_list`.
    """
    results = {}
    with tqdm(total=len(args_list)) as progress_bar:
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures_dict = {executor.submit(f, **args): ind for ind, args in enumerate(args_list)}
            for future in futures.as_completed(futures_dict):
                ind = futures_dict[future]
                try:
                    results[ind] = future.result()
                except Exception as e:
                    if return_exceptions:
                        results[ind] = e
                    else:
                        raise
                progress_bar.update(1)

    # Reorders the results to be in the same order as the input
    results = [results[ind] for ind in range(len(results))]
    return results