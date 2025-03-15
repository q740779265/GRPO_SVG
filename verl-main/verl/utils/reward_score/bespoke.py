from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def compute_score(data_source, solution_str, ground_truth) -> float:
    try:
        # 确保输入是字符串类型
        if not isinstance(solution_str, str) or not isinstance(ground_truth, str):
            return 0.0
        answer_parsed = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # if not answer_parsed:
        #     print(f"Failed to parse solution: {solution_str}")
        gold_parsed = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
            extraction_mode="first_match",
        )
        if not gold_parsed:
            print(f"Failed to parse ground truth: {ground_truth}")

        # print('answer_parsed:', answer_parsed)
        # print('gold_parsed:', gold_parsed)
        # print('-'*40)
        if not answer_parsed or not gold_parsed:
            return 0.0
        return float(verify(answer_parsed, gold_parsed))
    except Exception as e:
        # print(e)
        return 0.0