import json
from langfuse.openai import openai
from dotenv import load_dotenv
import functools
import os
import glob
from GuidelineHandler import parse_guidelines, guideline_to_string, split_guidelines_into_sections
import logging
from operator import itemgetter
from LatexParser import (
    get_document_structure_latex,
    convert_to_json,
    process_zip_file,
    get_document_section_content_map,
    get_all_section_titles_latex,
    clean_latex_content,
)
import commonmark
from pdf_parser.process_pdf import process_pdf_task
from src.agent.base import Agent
from src.context import ConcurrentTaskManager

CONCURRENCY_WORKERS = 16


# return category, explanation, confidence
def classify_paper(client, paper_title, model='gpt-4o-mini', paper_abstract="", whole_content=""):

    prompt = f"""You are an expert paper classifier. Given either a): the **title** and **abstract** of a paper, or b): the **title** and **whole content** of a paper, your task is to classify the paper into the most appropriate category from the list of paper types described. Each category includes a brief description to guide your classification. For both inputs, if either the title and abstract, or the title and whole content are insufficient to make a confident classification, please specify that and explain why.

---

**Paper Types:**

1. **Survey Paper (Review Paper):**
   - Provides a comprehensive overview of the literature on a specific topic.
   - Summarizes and synthesizes existing research, highlighting trends, gaps, and potential future directions.

2. **System Paper:**
   - Describes the design, implementation, cost analysis and modeling and evaluation of a solid prototype system or framework.
   - Focuses on performance, scalability, practical deployment, and technical challenges.
   - For systems that are not from industry (academic research systems).
   - Compared with a demo paper, a system paper is a long paper, and usually does not have a demo plan or user interaction.

3. **Algorithm Paper:**
   - Introduces new algorithms or significant improvements to existing ones, or introduces a new AI model, framework, or architecture designed for specific types of tasks or data.
   - Emphasizes algorithmic design, theoretical analysis, complexity, and proofs of correctness or bounds.

4. **Demo Paper:**
   - Showcases a working prototype or demonstration of a tool or application.
   - Focuses on practical demonstration, user interaction, and the demonstration plan.
   - Compared with a system paper, a demo paper is a short paper, and has more focus on the demo plan and user interaction, whereas a system paper normally does not have a demo plan or user interaction.

5. **Vision Paper:**
   - Presents forward-looking ideas or proposes new research directions, or presents an argument or perspective on a particular issue within the field.
   - Explores high-level concepts, usually without detailed experiments.
   - Aims to spark discussion or debate, providing supporting arguments and critical analysis.

6. **Short Paper (Work-in-Progress Paper):**
   - Provides preliminary results or ongoing research not yet mature enough for a full paper.
   - Focuses on motivation, methodology, and early findings.

7. **Tutorial Paper:**
   - Offers instructional content on a specific topic, aimed at educating the audience.
   - Includes comprehensive background information, explanations, and practical applications.

8. **Application Paper:**
   - Focuses on the practical application of existing theories or methods to solve real-world problems.
   - Describes problem context, adaptation of methods, results, and impact assessment.

9. **Data Paper:**
   - Describes datasets made available to the research community.
   - Includes data collection methods, structure, validation, and potential uses.
   - The difference to "Benchmark Paper (Dataset)" is that the data paper usually does not have performance baselines for specific tasks.

10. **Benchmark Paper (Dataset)**:
   - Introduces new datasets, with a focus on dataset collection, annotation, validation, and performance baselines for specific tasks.
   - Compared with a data paper, a benchmark paper has more experiments on baselines and findings.
   - Compared with an experimental survey paper, a benchmark paper focuses on new datasets.

11. **Experiment, Analysis and Benchmark Paper**:
   - Revisits and conducts a comprehensive study and empirical comparison of existing algorithms or methods on established tasks, often evaluating strengths, weaknesses, and performance across various conditions.
   - Typically does not include new datasets but focuses on existing algorithms or methods.

12. **Industry Track Paper:**
   - Covers all aspects of innovative commercial or industrial-strength systems and solutions.
   - Welcomes novel applications of systems and experiences in applying recent research advances to problems relevant to the industry.
   - Must articulate the innovative aspect of a product or application project (including relevant open-source software) and should not provide only a general overview or address a pure research problem.

13. **Research Paper (Technical Paper)**:
   - Presents original research findings, new methodologies, algorithms, or theoretical advancements.
   - Focuses on technical innovation and empirical validation.
   - **Default** if none of the above types apply.

14. **Rebuttal and Response Paper**:
   - Presents a rebuttal or response to a reviewer's comments.
   - Focuses on addressing the concerns raised in reviewer's comments.

15. **Blog Post Paper**:
   - Presents a blog post on a specific topic.
   - Focuses on providing information and insights on a particular topic. The example topics could be: 1) Reviews past work and summarizes outcomes, develops new intuitions, or highlights shortcomings. 2) Presents novel perspectives or interpretations of existing machine learning concepts or techniques. 3) Discusses important issues in machine learning, such as reproducibility, from a novel perspective. 4) Analyzes the societal implications of recent advancements in machine learning and AI. 5) Shares cool research ideas that were tried but did not work out.

**Instructions:**

- **Classification:** Identify the most appropriate category for the paper based on the definitions above.
- **Explanation:** Provide a brief explanation for your choice, citing specific elements from the title and abstract that support your classification.
- **Confidence:** The confidence level of your assessment on a scale of 1 to 5:
- 1 means extremely unconfident (less than 20% chance of being correct),
- 2 means low confidence (20-40% chance correct),
- 3 means moderate confidence (about 50% chance correct),
- 4 means high confidence (over 80% chance correct),
- 5 means extremely confident (over 90% chance correct and very likely to be accurate)


--- the title begins
{paper_title}
--- the title ends

--- the abstract of the paper begins
{paper_abstract}
--- the abstract of the paper ends

--- the whole content of the paper begins
{whole_content}
--- the whole content of the paper ends

Please provide your response in the following format:
--- your category starts
Category: [the category you will give]
--- your category ends

--- your explanation starts
[Your explanation for your assessment]
--- your explanation ends

--- your confidence starts
Confidence: [the confidence level of your assessment]
--- your confidence ends
"""

    agent = Agent("classify_paper", tags=["PaperScore"])
    result = agent.run(
        prompt=[
            {"role": "system", "content": "You are an expert academic paper reviewer and editor."},
            {"role": "user", "content": prompt},
        ],
        model=model
    )

    # Extract the revised section and explanation
    categoryText = result.split("--- your category starts")[1].split("--- your category ends")[0].strip()
    # scoreText is like "Score: 4"
    category = categoryText.split(":")[1].strip()
    explanation = result.split("--- your explanation starts")[1].split("--- your explanation ends")[0].strip()
    confidenceText = result.split("--- your confidence starts")[1].split("--- your confidence ends")[0].strip()
    # scoreText is like "Score: 4"
    confidence = confidenceText.split(":")[1].strip()
    return category, explanation, confidence


def paper_revision_suggestion_criteria(client, paper_latex, paper_type, criteria, score, explanation, model='gpt-4o-mini'):

    print(f"    Revising paper for criteria: {criteria['category']}")
    revision = ""
    # if score >=4:
    #    print(f"score is 4 or 5, no need to revise\n")
    # return revision

    # otherwise, revise the paper
    prompt = f"""You are a detail-oriented, expert editor specializing in academic computer science papers. Your task is to propose high-level structural and conceptual revisions to the paper so it achieves a top score of 5. Focus on addressing the weaknesses highlighted in the review’s explanation. Rather than line-by-line edits, provide a step-by-step plan on how to improve entire sections, add necessary details, and introduce clarity in methodology, data annotation protocols, data cleaning steps, and overall coherence.

**Instructions:**
- Carefully read the review’s explanation and the paper’s current state.
- Identify major sections or subsections where changes can be made to address the reviewer’s concerns.
- Provide up to three high-level revisions, each focusing on a broader aspect of the paper:
  - For each revision, describe the current gap or problem.
  - Suggest how to restructure or expand the relevant section. For example:
    - Introduce a new subsection that comprehensively details the annotation steps and criteria.
    - Add a flowchart, diagrams, or tables to better communicate the data annotation and cleaning process.
    - Integrate a more rigorous explanation of the tools and scripts used.
    - Clearly define filtering criteria, rationales, and data validation steps.
  - Use placeholders for sections or figures, rather than just a sentence or two.
  - Explain how your proposed revision addresses the reviewer’s concerns and raises the paper to a top standard.

By focusing on high-level changes, your proposed revisions should guide the authors in substantially improving the paper’s structure, clarity, and methodology presentation, rather than offering superficial line-by-line edits.

--- the content of the paper starts
{paper_latex}
--- the content of the paper ends

--- score scale and review starts
Criteria: {criteria['category']}.{criteria['prompt']}
Score (out of 5): {score}
Score Weightage (out of 100): {criteria['score_weightage']}
**Explanation:** {explanation}
--- score scale and review ends

--- output instructions starts
Please provide the step-by-step plan for high-level revisions only. Do not include the original introductory paragraph beginning like 'To address the reviewer's concerns...' and the concluding paragraph starting like 'By addressing these high-level structural and conceptual revisions...'. Only include the main recommended structural and conceptual improvements without the first and last paragraphs.

--- output instructions ends"""

    agent = Agent("paper_revision_suggestion_criteria", tags=["PaperScore"])
    revision = agent.run(
        prompt=[
            {"role": "system", "content": "You are an expert academic paper reviewer and editor."},
            {"role": "user", "content": prompt},
        ],
        model=model
    )
    return revision


def evaluate_criteria_on_paper(client, paper_latex, paper_type, criteria, model='gpt-4o-mini'):

    prompt = f"""You are an expert, responsible, and critical reviewer in the field of computer science, having reviewed many academic papers. You are tasked with evaluating a {paper_type} thoroughly and providing a critical and detailed review. Specifically, you will:

1. Score the paper based on the following criteria:
{criteria['prompt']}
   - Provide a single score from 0 to 400 that best reflects the quality of the paper for this criterion.
   - Base your score on the given scoring scale and the actual content of the paper.
2. Provide actionable and specific comments suggesting how to improve the paper in relation to this criterion:
   - Your comments should be constructive, pointing out both strengths and areas in need of improvement.
   - Refer explicitly to details or sections of the paper).
   - Provide improvement suggestions in a way that would be genuinely helpful to the authors to improve the paper according to the criterion.

Please thoroughly review the paper content provided and produce detailed suggestions that reflect careful reading and analysis of the paper.

--- score scale begins
Each criterion is scored on a scale from **0 to 400**, with increments of 1 in between. Here is the general score pivot for your reference:
- **0 (Poor)**: Significant issues; needs substantial improvement.
- **100 (Fair)**: Below average; several areas need enhancement.
- **200 (Good)**: Top 50% of the papers but not top 20%; Adequate but room for improvement.
- **300 (Very Good)**: Top 20% of the papers but not top 5%; minor improvements needed.
- **400 (Excellent)**: Top 5% of the papers; meets or exceeds top computer science paper standards.
--- score scale ends

--- the paper content starts
{paper_latex}
--- the paper content ends

Please provide your response in the following format:

--- your score starts
Score: [The score you assign based on the criteria]
--- your score ends

--- Actionable suggestions for improvement starts
[Your specific, actionable explanations and improvement suggestions based on your assessment. For example:
- "Section 2.1 can be enhanced by providing more detail on the methodology used [Add more technical details and references]."
- "The results in Table 3 need clarification on how the metrics were computed [Explain the computation in detail]."
- "The introduction should better contextualize the problem by referencing related works in the field [Add references to relevant studies]."
]
--- Actionable suggestions for improvement ends
"""

    agent = Agent("evaluate_criteria_on_paper", tags=["PaperScore"])
    result = agent.run(
        prompt=[
            {"role": "system", "content": "You are an expert academic paper reviewer and editor."},
            {"role": "user", "content": prompt},
        ],
        model=model
    )

    # Extract the revised section and explanation
    scoreText = result.split("--- your score starts")[1].split("--- your score ends")[0].strip()
    # scoreText is like "Score: 4"
    score = scoreText.split(":")[1].strip()
    explanation = (
        result.split("--- Actionable suggestions for improvement starts")[1]
        .split("--- Actionable suggestions for improvement ends")[0]
        .strip()
    )

    return score, explanation

# 不要有ADJACENT PARAGRAPH,用户在训练时并不会特地给出，是需要模型意识的
def provide_improvements_on_paper(client, paper_latex, paper_type, criteria, section, model='gpt-4o-mini'):
    # 定义角色和指令
    roles = [
        {
            "role": f"We have a paper improvement task with a specific criteria '{criteria['prompt']}'. Now play a role as an author of the provided paper content. Select a specific content from the section '{section}' (or equivalent), and ask a chatbot assistant to help you improve that selected content",
            "instr": [
                f"The selected paper content must be a worth-improving paragraph(s) that might not achieve the standards of the criteria '{criteria['prompt']}', and that content should come from the section '{section}' (or equivalent). The selected content will be labeled as **BEFORE IMPROVEMENT**.",
                # "The paragraphs before and after the selected content to provide more context information to the assistant. They are labeled as **ADJACENT PARAGRAPH**.",
                "Provide a concise, conversational improvement-related question labeled as **QUESTIONS**. These questions should not explicitly tell what rules or standards to follow or what the specific goal should be. Instead, offer a high-level instruction that may hint at the criteria without stating them directly. The aim is to allow for creativity and subtle alignment with the criteria.",
                "Keep the question short and conversational."
            ],
            "outputs": [
#                 """
# --- ADJACENT PARAGRAPH (BEFORE) STARTS  
# <The adjacent paragraph before the selected content. Output NONE if the selected content is the first paragraph of the section.>
# --- ADJACENT PARAGRAPH (BEFORE) ENDS  
#                 """,
                """
--- BEFORE IMPROVEMENT STARTS  
<Selected content>
--- BEFORE IMPROVEMENT ENDS  
                """,
#                 """
# --- ADJACENT PARAGRAPH (AFTER) STARTS  
# <The adjacent paragraph after the selected content. Output NONE if the selected content is at the end of the section.>
# --- ADJACENT PARAGRAPH (AFTER) ENDS  
#                 """,
                f"""
--- QUESTIONS START  
<Concise, improvement-related question based on the criteria '{criteria['prompt']}'>
--- QUESTIONS END  
                """
            ]
        },
        {
            "role": "Act as an expert model for improving articles.",
            "instr": [
                "The revised version of the selected content should be labeled as AFTER IMPROVEMENT and specifically address the QUESTIONS on BEFORE IMPROVEMENT above. Avoid adding unnecessary length, unrelated details, overclaims, or vague statements. Focus on clear, concise, and evidence-based improvements that align with the overall context of the paper.",
                "Provide a detailed explanation of the changes made, labeled as EXPLANATION, with clear references to the paper's content. Ensure the explanation demonstrates how the revisions align with the context and criteria of the paper."
            ],
            "outputs": [
                """
--- AFTER IMPROVEMENT STARTS  
<Revised version of the selected content to answer the **Questions** above>
--- AFTER IMPROVEMENT ENDS  
                """,
                """
--- EXPLANATION STARTS  
<An explanation of the changes made, showing how they align with the context of the article and address the criteria. Include references from the paper context where relevant.>
--- EXPLANATION ENDS  
                """
            ]
        }
    ]

    # 构建完整的 prompt
    system_prompt = """
You are an advanced language model designed to assist users in improving their articles. Users will provide an article in LaTeX or Markdown format and specify a **section** along with **criteria** for improvement. Your task is to identify a specific selected content from the provided section, align it with the given criteria, and offer actionable feedback to improve the content.
"""

    # 构建指令部分
    instructions_prompt = "### Instructions:\n"
    for idx, r in enumerate(roles):
        role, instrs, outputs = r['role'], r['instr'], r['outputs']
        instructions_prompt += f"{idx+1}. **Role {idx+1}**: {role}\n"
        for instr in instrs:
            instructions_prompt += f"   - {instr}\n"

    # print(f"paper_latex[:100]: {paper_latex[:100]}")
    # 构建论文内容部分
    paper_prompt = f"""
--- PAPER CONTEXT STARTS
{paper_latex}
--- PAPER CONTEXT ENDS
"""

    # 构建输出格式部分
    format_prompt = "### Response Format (must be strictly followed):\n"
    all_outputs = []
    for idx, r in enumerate(roles):
        all_outputs += r['outputs']
    format_prompt += "\n".join(all_outputs)

    # 组合所有部分
    combined_prompt = f"{system_prompt}\n{instructions_prompt}\n{paper_prompt}\n{format_prompt}"

    # 调用 Agent
    agent = Agent("evaluate_and_suggest_improvements", tags=["PaperScore"], model=model)
    # print("model: ", model)
    if 'o1' in model:
        result = agent.run(
            prompt=[
                {"role": "user", "content": combined_prompt},
            ]
        )
    else:
        result = agent.run(
            prompt=[
                {"role": "system", "content": "You are a professional academic researcher"},
                {"role": "user", "content": combined_prompt},
            ]
        )
    print(f"result: {result}")
    # 提取改进内容
    # print("model: ", model)
    # # 提取相邻段落（之前）
    # adjacent_before = result.split("--- ADJACENT PARAGRAPH (BEFORE) STARTS")[1].split("--- ADJACENT PARAGRAPH (BEFORE) ENDS")[0].strip()
    
    # 提取需要改进的内容
    before_improvement = result.split("--- BEFORE IMPROVEMENT STARTS")[1].split("--- BEFORE IMPROVEMENT ENDS")[0].strip()
    
    # # 提取相邻段落（之后）
    # adjacent_after = result.split("--- ADJACENT PARAGRAPH (AFTER) STARTS")[1].split("--- ADJACENT PARAGRAPH (AFTER) ENDS")[0].strip()
    
    # 提取问题
    questions = result.split("--- QUESTIONS START")[1].split("--- QUESTIONS END")[0].strip()
    
    # 提取改进后的内容
    after_improvement = result.split("--- AFTER IMPROVEMENT STARTS")[1].split("--- AFTER IMPROVEMENT ENDS")[0].strip()
    
    # 提取解释
    explanation = result.split("--- EXPLANATION STARTS")[1].split("--- EXPLANATION ENDS")[0].strip()

    return before_improvement, questions, after_improvement, explanation

# a function to show all the scores and explanations in a HTML file
def show_scores_and_explanations_revisions(scores_and_explanations, paper_title, paper_type, output_directory):
    html_content = "<html><body><h1>Scores and Explanations</h1>"
    # give a summary of categories and scores
    html_content += "<h2>Summary of Categories and Scores</h2>"
    total_score = 0
    full_score = 0
    # output the paper_title and paper_type
    html_content += f"<h2>Paper Title: {paper_title}</h2>"
    html_content += f"<h2>Paper Type: {paper_type}</h2>"
    print(f"\n\nhtml-size of scores_and_explanations: {len(scores_and_explanations)}\n\n")
    for criteria, score, explanation, revision in scores_and_explanations:
        total_score += score * int(criteria["score_weightage"])
        full_score += 5 * int(criteria["score_weightage"])

    if full_score > 0:
        html_content += f"<p>Percentage (out of 100): {int(total_score/full_score*100)}%</p>"
    else:
        html_content += f"<p>Percentage (out of 100): Not available</p>"
    html_content += "<hr>"
    # give a table to show the scores, score weightage and score.
    html_content += f"<p style='color: red;'>Pay special attention to the criteria with scores less than 4. Sample revisions are given to improve the paper for these criteria.</p>"
    html_content += (
        "<table border='1'><tr><th>Criteria</th><th>Score Weightage (out of 100)</th><th>Score (out of 5)</th></tr>"
    )
    # add a row for each criteria
    for criteria, score, explanation, revision in scores_and_explanations:
        if score < 4:
            html_content += f"<tr><td>{'<strong>' + criteria['category'] + '</strong>'}</td><td>{'<strong>' +criteria['score_weightage']+ '</strong>'}</td><td>{'<strong>' + str(score) + '</strong>'}</td></tr>"
        else:
            html_content += (
                f"<tr><td>{criteria['category']}</td><td>{criteria['score_weightage']}</td><td>{score}</td></tr>"
            )
    html_content += "</table>"

    for criteria, score, explanation, revision in scores_and_explanations:
        total_score += score
        full_score += int(criteria["score_weightage"])
        html_content += f"<h2>Criteria: {criteria['category']}</h2>"
        # html_content += f"<p style='font-weight: normal;'>Description: {criteria['prompt']}</p>"
        html_content += (
            f"<p style='font-weight: normal;'>Score Weightage (out of 100): {criteria['score_weightage']}</p>"
        )
        html_content += (
            f"<p><strong>Score (out of 5): {score}</strong></p>" if score < 4 else f"<p>Score (out of 5): {score}</p>"
        )
        explanation_html = commonmark.commonmark(explanation)
        html_content += f"<p style='font-weight: normal;'>**Explanation:** {'<strong>' + explanation_html + '</strong>' if score < 4 else explanation_html}</p>"
        # Convert the revision from Markdown to HTML
        revision_html = commonmark.commonmark(revision)
        if score < 4:
            html_content += (
                f"<p style='font-weight: normal;'>**Revision:** {'<strong>' + revision_html + '</strong>'}</p>"
            )
        else:
            html_content += f"<p style='font-weight: normal;'>**Revision:** {revision_html}</p>"
    html_content += "</body></html>"
    # show the scores and explanations in a HTML file
    with open(f"{output_directory}/scores_and_explanations.html", "w", encoding="utf-8") as f:
        f.write(html_content)


def evaluate_and_suggest_revision(client, paper_content, paper_type, criteria, output_directory, model='gpt-4o-mini'):
    """Combines evaluation and revision suggestion into a single function"""
    print(f"    Evaluating paper for criteria: {criteria['category']}")
    cache_file = f"{output_directory}/cache.paper_score_evaluate_criteria_{criteria['category']}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            score, explanation = json.load(f)
    else:
        score, explanation = evaluate_criteria_on_paper(client, paper_content, paper_type, criteria)
        with open(cache_file, "w") as f:
            json.dump({"score": score, "explanation": explanation}, f, indent=4)

    # cache_file = f"{output_directory}/cache.paper_score_revision_criteria_{criteria['category']}.json"
    # if os.path.exists(cache_file):
    #     with open(cache_file, "r") as f:
    #         revision = json.load(f)
    # else:
    #     revision = paper_revision_suggestion_criteria(client, paper_content, paper_type, criteria, score, explanation)
    #     with open(cache_file, "w") as f:
    #         json.dump({"revision": revision}, f, indent=4)

    return criteria, score, explanation, None

def evaluate_and_suggest_improvement(client, paper_content, paper_type, criteria, section,output_directory, model='gpt-4o-mini'):
    """Combines evaluation and improvement suggestion into a single function"""
    print(f"   Improving paper for criteria: {criteria['category']}")
    cache_file = f"{output_directory}/cache.paper_score_evaluate_criteria_{criteria['category']}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            before_improvement, questions, after_improvement, explanation = json.load(f)
    else:
        before_improvement, questions, after_improvement, explanation = provide_improvements_on_paper(client, paper_content, paper_type, criteria, section, model=model)
        with open(cache_file, "w") as f:
            json.dump({ "before_improvement": before_improvement, "questions": questions, "after_improvement": after_improvement, "explanation": explanation}, f, indent=4)

    return before_improvement, questions, after_improvement, explanation


def Score_Paper(
    client, output_directory, model='gpt-4o-mini', paper_title="", paper_abstract="", paper_whole_content="", sample_paper_criteria_path=""
):
    # load the criteria
    criteria_array = []
    criteria_file = "paper-score-criteria.json"
    if not os.path.exists(criteria_file):
        print(f"Criteria file {criteria_file} does not exist")
        return
    with open(criteria_file, "r") as f:
        criteria_array = json.load(f)
    print(f"Loaded criteria: {criteria_array.keys()}")

    # if sample_paper_criteria_path is provided, load it and attach to criteria_array as "Sample Paper"
    if sample_paper_criteria_path:
        with open(sample_paper_criteria_path, "r") as f:
            sample_criteria_array = json.load(f)
        criteria_array["Sample Paper"] = sample_criteria_array

    scores_and_explanations_revisions = []
    scoreJsonFile = f"{output_directory}/paper_score_scores_and_explanations.json"
    paperTypeJsonFile = f"{output_directory}/paper_score_paper_type.json"
    paper_type = ""
    # check the paper type
    if not os.path.exists(paperTypeJsonFile):
        if paper_abstract == "":
            paper_type, explanation, confidence = classify_paper(client, paper_title, "", paper_whole_content)
        else:
            paper_type, explanation, confidence = classify_paper(client, paper_title, paper_abstract, "")
        print(
            f"\n\nObtain from GPT, Paper type: {paper_type}, Paper title: {paper_title}, Explanation: {explanation}, Confidence: {confidence}"
        )
        with open(paperTypeJsonFile, "w") as f:
            json.dump(
                {
                    "paper_type": paper_type,
                    "paper_title": paper_title,
                    "explanation": explanation,
                    "confidence": confidence,
                },
                f,
                indent=4,
            )
    else:
        with open(paperTypeJsonFile, "r") as f:
            paper_type_info = json.load(f)
            paper_type = paper_type_info["paper_type"]
            paper_title = paper_type_info["paper_title"]
            explanation = paper_type_info["explanation"]
            confidence = paper_type_info["confidence"]
            print(f"loaded paper type from json file:{paperTypeJsonFile}")
            print(
                f"\n\nPaper type: {paper_type}, Paper title: {paper_title}, Explanation: {explanation}, Confidence: {confidence}"
            )

    # Final criteria array: the union of criteria_array (from file paper-score-criteria.json) and the criteria extracted from the sample papers
    # If paper_type is not in criteria_array:
    #   If sample paper provided: set paper_type as "Sample Paper" and use criteria_array["Sample Paper"] as final_criteria;
    #   If no sample papers: set paper_type as "Research Paper (Technical Paper)" and use criteria_array["Research Paper (Technical Paper)"] as final_criteria;
    final_criteria = []

    if paper_type not in criteria_array:
        if "Sample Paper" in criteria_array and sample_paper_criteria_path:
            print(f"\n\nPaper type {paper_type} not found in criteria_array, using Sample Paper criteria\n\n")
            paper_type = "Sample Paper"
            final_criteria = criteria_array["Sample Paper"]
        else:
            print(
                f"\n\nPaper type {paper_type} not found in criteria_array, also no sample papers provided, using Research Paper (Technical Paper) criteria\n\n"
            )
            paper_type = "Research Paper (Technical Paper)"
            final_criteria = criteria_array["Research Paper (Technical Paper)"]
    else:
        print(f"\n\nPaper type {paper_type} found in criteria_array\n\n")
        final_criteria = criteria_array[paper_type]
        if "Sample Paper" in criteria_array and sample_paper_criteria_path:
            existing_categories = {c["category"] for c in final_criteria}

            # Add sample paper criteria that don't already exist
            for sample_criteria in criteria_array["Sample Paper"]:
                if sample_criteria["category"] not in existing_categories:
                    final_criteria.append(sample_criteria)
                    existing_categories.add(sample_criteria["category"])

    if os.path.exists(scoreJsonFile):
        with open(scoreJsonFile, "r") as f:
            scores_and_explanations_revisions = json.load(f)
            print(f"Loaded scores and explanations from {scoreJsonFile}")
    else:
        with ConcurrentTaskManager(max_workers=CONCURRENCY_WORKERS) as manager:
            for criteria in criteria_array[paper_type]:
                print(f"Submit evaluating and revising task for criteria: {criteria['category']}")
                manager.submit_task(evaluate_and_suggest_revision, client, paper_whole_content, paper_type, criteria, output_directory, model=model)
            print("Waiting for all tasks to complete")
            results = manager.get_results()
            for criteria, score, explanation, revision in results:
                print(f"    Retrieved results for criteria: {criteria['category']}")
                scores_and_explanations_revisions.append((criteria, score, explanation, revision))

            with open(scoreJsonFile, "w") as f:
                json.dump(scores_and_explanations_revisions, f, indent=4)
                print(f"Saved scores and explanations to {scoreJsonFile}")
    # show the scores and explanations in a HTML file
    show_scores_and_explanations_revisions(
        scores_and_explanations_revisions, paper_title, paper_type, output_directory
    )

    # remove all the cache files
    for f in glob.glob(os.path.join(output_directory, "cache.*")):
        os.remove(f)

def main():
    # Load environment variables
    load_dotenv()

    # Set up logging
    logging.basicConfig(
        filename="PaperScore.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'")
        return

    # Create OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    file_dir = "../input-latex"
    output_directory = "rule_output"
    # sample_paper_criteria_path = "paper_criteria/output/blogpost_criteria.json"
    sample_paper_criteria_path = ""
    files = [f for f in os.listdir(file_dir) if not f.startswith(".") and os.path.isfile(os.path.join(file_dir, f))]
    for file in files:
        unique_output_directory = ""
        paper_title = ""
        paper_abstract = ""
        paper_whole_content = ""

        if file.endswith(".zip"):
            full_zipfile_path = os.path.join(file_dir, file)
            print(f"Processing zipfile: {full_zipfile_path}")

            # Extract the base name of the zip file (without extension)
            base_name = os.path.splitext(os.path.basename(full_zipfile_path))[0]

            # Create a unique output directory based on the zip file name
            unique_output_directory = f"{output_directory}/{base_name}"
            os.makedirs(unique_output_directory, exist_ok=True)

            paper_latex_content_original = process_zip_file(full_zipfile_path, unique_output_directory)
            paper_whole_content = clean_latex_content(paper_latex_content_original)

            document_structure = get_document_structure_latex(paper_whole_content)
            paper_title = document_structure["title"]
            paper_abstract = document_structure["abstract"]
            print(f"\n\nPaper title: {paper_title}\n\nPaper abstract: {paper_abstract}")

        elif file.endswith(".pdf"):
            full_pdf_path = os.path.join(file_dir, file)
            print(f"Processing pdf file: {full_pdf_path}")

            base_name = os.path.splitext(os.path.basename(full_pdf_path))[0]
            unique_output_directory = f"{output_directory}/{base_name}"
            os.makedirs(unique_output_directory, exist_ok=True)

            results = process_pdf_task(file_path=full_pdf_path, output_path=unique_output_directory)
            paper_title = results["structured_data"]["title"]
            if "abstract" in results["structured_data"]:
                paper_abstract = results["structured_data"]["abstract"]
            else:
                paper_abstract = ""
            paper_whole_content = results["full_text"]
        else:
            print(f"Unknown file type: {file}")
            continue

        Score_Paper(
            client,
            unique_output_directory,
            paper_title,
            paper_abstract,
            paper_whole_content,
            sample_paper_criteria_path,
        )
        print(f"OK done file: {file}")


if __name__ == "__main__":
    print = functools.partial(print, flush=True)
    main()
