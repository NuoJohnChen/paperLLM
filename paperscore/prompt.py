from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
# 模型和分词器路径
model_path = "/disk1/nuochen/models/Qwen2-72B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配 GPU
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()



# 配置 DeepSpeed
ds_config = {
    "train_micro_batch_size_per_gpu": 1,  # 每个 GPU 的 batch size
    "gradient_accumulation_steps": 1,     # 梯度累积步数
    "fp16": {                             # 启用 FP16 混合精度
        "enabled": True
    },
    "zero_optimization": {                # 启用 ZeRO 优化
        "stage": 3,                       # ZeRO 优化级别 (3 是最高级别)
        "offload_param": {                # 参数分布到 CPU 或 NVMe
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,             # 通信与计算重叠
        "contiguous_gradients": True,     # 连续的梯度内存
    },
    "zero_allow_untested_optimizer": True,
    "distributed_backend": "nccl",
    "gradient_checkpointing": True,
}

model = deepspeed.initialize(model=model, config=ds_config)[0]
# # # 检查是否有可用的 GPU
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # 将模型移动到 GPU
# # model = model.to(device)

# # 定义 prompt
# # prompt = (
# #     "You are an advanced LLM designed to assist in improving articles. Users will send you a PDF of their article and select specific paragraphs for improvement-related questions. "
# #     "Each paragraph's improvement is evaluated based on the following criteria. For each criterion, generate two potential user questions that aim to improve the article. "
# #     "Your questions should guide revisions that directly enhance the article, rather than merely assessing its quality. The criteria are as follows:"
# # )

# # prompt= """
# # You are an advanced language model designed to assist users in improving their articles. Users will provide an article in Markdown format and select a specific excerpt to ask improvement-related questions.
# # Improvement questions should align with the given standards and guide users to modify a particular text, directly enhancing the quality of the article rather than merely assessing its strengths and weaknesses.

# # First, assume you are the user and provide the selected "excerpt" along with improvement questions based on the standards. Then, assume the role of an expert model for improving articles and generate the corresponding revised version of the text.
# # The revised answer should be based on the context of the article, and you should also provide an explanation of the changes made in relation to the context.
# # """

# prompt= """
# You are an advanced language model designed to assist users in improving their articles. Users will provide an article in Markdown format and select a specific excerpt to ask improvement-related questions.
# Your task is to provide constructive suggestions that align with given standards, helping users make targeted modifications to enhance the quality of the article. Avoid general assessments or excessive praise, and focus on actionable feedback specific to the selected excerpt.
# First, assume you are the user and provide the selected "excerpt" along with improvement questions based on the standards. Then, assume the role of an expert model for improving articles and generate the corresponding revised version of the text.
# The revised answer should take into account the context of the entire article, referencing relevant sections where necessary, and include an explanation of the changes made in relation to the overall context.
# """
# # prompt += """
# # \begin{table*}[p]
# #     \centering
# #     \caption{Criteria of Introduction}
# #     \label{tab:commentsdetails}
# #     \resizebox{\linewidth}{!}{
# #     \begin{tabular}{lp{0.4\linewidth}}
# #         \toprule
# #         \textbf{Aspect} & \textbf{Comments} \\
# #         % \toprule
# #          \hline 
# #         \multirowcell{6}{Clear and Comprehensive Problem Definition and Motivation} &Clarity of Problem Definition\\
# #         & Motivation and Significance \\
# #         & Literature Review \\
# #         & Experimental Validation \\
# #         & Visual Aids \\
# #         & Technical Depth \\

# #         \bottomrule

# #     \end{tabular}}
# # \end{table*}
# # """

# prompt += """
# --- the criteria starts
# \begin{table*}[p]
#     \centering
#     \caption{Criteria of Introduction}
#     \label{tab:commentsdetails}
#     \resizebox{\linewidth}{!}{
#     \begin{tabular}{lp{0.4\linewidth}}
#         \toprule
#         \textbf{Aspect} & \textbf{Comments} \\
#         % \toprule
#          \hline 
#         \multirowcell{6}{Clear and Comprehensive Problem Definition and Motivation} &Clarity of Problem Definition\\
#         & Motivation and Significance \\
#         & Literature Review \\
#         & Experimental Validation \\
#         & Visual Aids \\
#         & Technical Depth \\

#         \bottomrule

#     \end{tabular}}
# \end{table*}
# --- the criteria ends
# """

# prompt += """
# --- the paper content starts
# A Differentiable Neural Computer (DNC) is a memory-augmented neural computation model capable of learning to solve complex algorithmic tasks, from simple sorting algorithms, through graph problems, to text question answering. In previous works, it was always given a constant number of planning steps to complete its task.\nIn this work, we argue that the number of planning steps the model is allowed to take, which we call \"planning budget\", is a constraint that can cause the model to generalize poorly and hurt its ability to fully utilize its external memory.\nBy introducing an adaptive planning budget that scales with input size during training, the model is better able to utilize its memory space, and achieves substantially better accuracy on input sizes not seen during training.\nWe experiment with Graph Shortest Path search, which has been used as a benchmark to measure these models in the past, and with the Graph MinCut problem.\nIn both problems, our proposed approach improves performance and generalizes better compared to the standard planning budget.\n\n## 1 Introduction\n\nIn the past few years, Deep Neural Networks (DNNs) produced substantial advances in a myriad of fields, from image classification and segmentation, through audio analysis and speech recognition, to generation of art and text with human-like accuracy. Despite these advances, a significant problem still remains: generalization to unseen inputs. Since a DNN is trained by solving an optimization task over a training set, the network often achieves lower performance on inputs that are out of distribution to it's training set. This can be attributed to sparse input distributions, outliers, edge cases and more. In order to improve generalization, DNNs are trained on ever increasing numbers of training samples. For example in NLP, dataset sizes can reach billions and trillions of tokens Kudugunta et al. (2023); Wang et al. (2023).\n\nA potential solution to the issue of generalization lies in algorithms, which usually are able to solve a problem for all cases. Instead of learning to approximate a function, we learn to produce an algorithm; a series of steps, changing internal state, memory, or external interfaces, that eventually reach the desired outcome. We can hope that if a good algorithm is found, it will generalize to all cases by definition. This idea is often called Algorithmic Reasoning, and we use this name in this paper.\n\nThere are multiple examples of algorithmic reasoning, which can be implemented in an explicit or an implicit manner. In the explicit approach, the model's task is to output a description of the algorithm it has learned. Examples include AlphaTensor Fawzi et al. (2022), in which the model learns to find general matrix multiplication algorithms for various matrix sizes; code generation models such as Li et al. (2022), and Large Language Model (LLM) that are able to generate a piece of code solving a task described in free text Shinn et al. (2023).\n\nIn the implicit approach, the processor learns to output actions that work for a problem instance. For example in graph shortest path, the input is a specific graph \\(G\\) and a query pair of source node and target node \\((s,t)\\). The correct output is the edges comprising the shortest path between \\(s\\) and \\(t\\). The processor's internal representation mimics a computer model, and its actions can be interpreted as instructions for processing memory or register values. However unlike the explicit approach, which outputs instructions that should work for all inputs, the implicit approach simply runs the learned algorithm on the given inputs. In order to run the algorithm, we must run the model. In this way, the model learns to perform the algorithm rather than describe it; the model's weights, internalrepresentation space, and architecture together comprise the learned algorithm. Examples include Zaremba and Sutskever (2016); Velickovic et al. (2020); Kurach et al. (2016); Graves et al. (2014)\n\nAn important example of this approach is the Differentiable Neural Computer model Graves et al. (2016), which is the focus of this work. In brief, the DNC is a Recurrent Neural Network (RNN) based on a differentiable implementation of a Turing Machine, extending previous work Graves et al. (2014). Featuring an LSTM with a memory matrix, the DNC can model algorithms that interact with external memory, handling tasks like copying, sorting, and graph problems. A summary of the DNC architecture is provided in Appendix G.\n\nIn short, The DNC processes inputs by iteratively absorbing a sequence of vectors, storing them in memory, and executing memory operations for task-specific outputs. It has several addressing mechanisms which allow it to allocate new cells, update existing ones orlookup a specific cell by its content. Its operation spans three phases: input, planning, and answering. Initially, it receives input, then undergoes \\(p\\) planning steps for processing--a number previously limited to zero or just 10 in more complex tasks--and finally produces the output in the answering phase.\n\nWe state and provide empirical evidence for two important claims:\n\n1. Choosing an adaptive planning budget allows the model to learn a more general algorithm than it can with a constant budget, allowing it to generalize to input sizes much larger than seen during training.\n2. Choosing an adaptive planning budget allows the model to learn better memory utilization, facilitating the use of larger memory for larger inputs.\n\nOur findings show that a DNC trained with a constant \\(p\\) faces limitations, most likely overfitting to heuristics and average case-attributes of the input. This issue persists unless it is trained with a substantially large \\(p\\), which also has inherent limitations. In contrast, the adaptive model avoids these issues, demonstrating more robust performance without the constraints observed in the constant budget models. The paper is structured as follows: Section 2 overviews related work; Section 3 details our method and its complexity theory motivation; Section 4 presents experimental analysis supporting our first claim; Section 5 discusses larger constant budgets and evidence for our second claim; and Section 6 concludes the paper.\n\n", "soundness": 1, "presentation": 2, "contribution": 2, "rating": 5, "confidence": 4}
# --- the paper content ends
# """



# prompt +="""
# Please respond in the following format:
# Before Improvement starts:
# Selected excerpt
# Before Improvement ends.

# Questions start:
# Selected excerpt with one improvement-related question based on the standards
# Questions end.

# After Improvement starts:
# Revised version of the excerpt
# After Improvement ends.

# Explanation starts:
# An explanation of the changes made, showing how they align with the context of the article and address the standards
# Explanation ends.
# """


prompt = f"""
You are an advanced language model designed to assist users in improving their articles. Users will provide an article in LaTeX or Markdown format and specify a **section** along with **criteria** for improvement. Your task is to identify a specific selected content from the provided section, align it with the given criteria, and offer actionable feedback to improve the content.

### Instructions:
1. **First Role**: Assume the role of the paper's author. Users will provide you with a simple, conversational instruction. Based on that instruction, select a specific selected content from the provided section, labeled as **Before Improvement**.
   - The selected paper content based on the criteria 'Contextual Relevance and Clarity of Background' should come from the section background, and will be labeled as **Before Improvement**.  
   - Provide a concise, conversational improvement-related question labeled as **Questions**. These questions should not explicitly tell the AI (you) what rules or standards to follow or what the specific goal should be. Instead, offer a high-level instruction that may hint at the criteria without stating them directly. The aim is to allow for creativity and subtle alignment with the criteria.
2. **Second Role**: Act as an expert model for improving articles. Provide:  
   - The improved version of the selected content labeled as **After Improvement**, designed to answer the **Questions** on **Before Improvement** above.  
   - A detailed explanation of the changes made, using **references from the paper context** to help answer the question and demonstrate alignment with the context and the criteria, labeled as **Explanation**.

--- the paper content starts  
A Differentiable Neural Computer (DNC) is a memory-augmented neural computation model capable of learning to solve complex algorithmic tasks, from simple sorting algorithms, through graph problems, to text question answering. In previous works, it was always given a constant number of planning steps to complete its task.\nIn this work, we argue that the number of planning steps the model is allowed to take, which we call \"planning budget\", is a constraint that can cause the model to generalize poorly and hurt its ability to fully utilize its external memory.\nBy introducing an adaptive planning budget that scales with input size during training, the model is better able to utilize its memory space, and achieves substantially better accuracy on input sizes not seen during training.\nWe experiment with Graph Shortest Path search, which has been used as a benchmark to measure these models in the past, and with the Graph MinCut problem.\nIn both problems, our proposed approach improves performance and generalizes better compared to the standard planning budget.\n\n## 1 Introduction\n\nIn the past few years, Deep Neural Networks (DNNs) produced substantial advances in a myriad of fields, from image classification and segmentation, through audio analysis and speech recognition, to generation of art and text with human-like accuracy. Despite these advances, a significant problem still remains: generalization to unseen inputs. Since a DNN is trained by solving an optimization task over a training set, the network often achieves lower performance on inputs that are out of distribution to it's training set. This can be attributed to sparse input distributions, outliers, edge cases and more. In order to improve generalization, DNNs are trained on ever increasing numbers of training samples. For example in NLP, dataset sizes can reach billions and trillions of tokens Kudugunta et al. (2023); Wang et al. (2023).\n\nA potential solution to the issue of generalization lies in algorithms, which usually are able to solve a problem for all cases. Instead of learning to approximate a function, we learn to produce an algorithm; a series of steps, changing internal state, memory, or external interfaces, that eventually reach the desired outcome. We can hope that if a good algorithm is found, it will generalize to all cases by definition. This idea is often called Algorithmic Reasoning, and we use this name in this paper.\n\nThere are multiple examples of algorithmic reasoning, which can be implemented in an explicit or an implicit manner. In the explicit approach, the model's task is to output a description of the algorithm it has learned. Examples include AlphaTensor Fawzi et al. (2022), in which the model learns to find general matrix multiplication algorithms for various matrix sizes; code generation models such as Li et al. (2022), and Large Language Model (LLM) that are able to generate a piece of code solving a task described in free text Shinn et al. (2023).\n\nIn the implicit approach, the processor learns to output actions that work for a problem instance. For example in graph shortest path, the input is a specific graph \\(G\\) and a query pair of source node and target node \\((s,t)\\). The correct output is the edges comprising the shortest path between \\(s\\) and \\(t\\). The processor's internal representation mimics a computer model, and its actions can be interpreted as instructions for processing memory or register values. However unlike the explicit approach, which outputs instructions that should work for all inputs, the implicit approach simply runs the learned algorithm on the given inputs. In order to run the algorithm, we must run the model. In this way, the model learns to perform the algorithm rather than describe it; the model's weights, internalrepresentation space, and architecture together comprise the learned algorithm. Examples include Zaremba and Sutskever (2016); Velickovic et al. (2020); Kurach et al. (2016); Graves et al. (2014)\n\nAn important example of this approach is the Differentiable Neural Computer model Graves et al. (2016), which is the focus of this work. In brief, the DNC is a Recurrent Neural Network (RNN) based on a differentiable implementation of a Turing Machine, extending previous work Graves et al. (2014). Featuring an LSTM with a memory matrix, the DNC can model algorithms that interact with external memory, handling tasks like copying, sorting, and graph problems. A summary of the DNC architecture is provided in Appendix G.\n\nIn short, The DNC processes inputs by iteratively absorbing a sequence of vectors, storing them in memory, and executing memory operations for task-specific outputs. It has several addressing mechanisms which allow it to allocate new cells, update existing ones orlookup a specific cell by its content. Its operation spans three phases: input, planning, and answering. Initially, it receives input, then undergoes \\(p\\) planning steps for processing--a number previously limited to zero or just 10 in more complex tasks--and finally produces the output in the answering phase.\n\nWe state and provide empirical evidence for two important claims:\n\n1. Choosing an adaptive planning budget allows the model to learn a more general algorithm than it can with a constant budget, allowing it to generalize to input sizes much larger than seen during training.\n2. Choosing an adaptive planning budget allows the model to learn better memory utilization, facilitating the use of larger memory for larger inputs.\n\nOur findings show that a DNC trained with a constant \\(p\\) faces limitations, most likely overfitting to heuristics and average case-attributes of the input. This issue persists unless it is trained with a substantially large \\(p\\), which also has inherent limitations. In contrast, the adaptive model avoids these issues, demonstrating more robust performance without the constraints observed in the constant budget models. The paper is structured as follows: Section 2 overviews related work; Section 3 details our method and its complexity theory motivation; Section 4 presents experimental analysis supporting our first claim; Section 5 discusses larger constant budgets and evidence for our second claim; and Section 6 concludes the paper.
--- the paper content ends  

### Response Format (must be strictly followed):

--- Before Improvement starts  
[Selected content]  
--- Before Improvement ends  

--- Questions start  
[Concise, improvement-related question based on the criteria 'Contextual Relevance and Clarity of Background' ]  
--- Questions end  

--- After Improvement starts  
[Revised version of the selected content to answer the **Questions** above]  
--- After Improvement ends  

--- Explanation starts  
[An explanation of the changes made, showing how they align with the context of the article and address the criteria. Include references from the paper context where relevant.]  
--- Explanation ends  
"""


# 编码输入并将其移动到 GPU
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# 生成回答
output = model.generate(input_ids, max_length=4000, num_return_sequences=1)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的回答
print(response)
