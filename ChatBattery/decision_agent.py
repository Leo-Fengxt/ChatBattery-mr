from .domain_agent import Domain_Agent


class Decision_Agent:
    @staticmethod
    def decide_one_pair(input_formula, output_formula, task_id):
        if task_id not in (101, 102):
            raise NotImplementedError(f"Unsupported task_id={task_id}. Expected 101 (Li) or 102 (Na).")

        input_value = Domain_Agent.calculate_theoretical_capacity(input_formula, task_id)
        try:
            output_value = Domain_Agent.calculate_theoretical_capacity(output_formula, task_id)
        except Exception:
            # If the candidate formula is malformed (e.g., "$PO_4"), treat it as invalid
            # and keep the UI/pipeline running.
            output_value = float("nan")
            return input_value, output_value, False

        return input_value, output_value, output_value > input_value * 1

    @staticmethod
    def decide_pairs(input_formula, output_formula_list, task_id):
        answer_list = []
        for output_formula in output_formula_list:
            input_value, output_value, answer = Decision_Agent.decide_one_pair(input_formula, output_formula, task_id)
            answer_list.append([output_formula, output_value, answer])
        return answer_list
