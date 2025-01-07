from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
            base_url="https://api.openai.com/v1",
            organization='org-Bis7Azo6YLHnUAacSoa4OVr0',
            project='proj_ABbJAUN09IIJrt8M7bkomA7T',
            api_key="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")

# 定义 prompt
prompt = """
You are an advanced language model designed to assist users in improving their articles. Users will provide an article in LaTeX or Markdown format and specify a **section** along with **criteria** for improvement. Your task is to identify a specific selected content from the provided section, align it with the given criteria, and offer actionable feedback to improve the content.

### Instructions:
1. **First Role**: Assume the role of the paper's author. Users will provide you with a simple, conversational instruction. Based on that instruction, select a specific selected content from the provided section, labeled as **Before Improvement**.
   - The selected paper content based on the criteria 'Clarity and Impact of Key Innovations and Findings' should come from the section conclusion, and will be labeled as **Before Improvement**.  
   - Provide a concise, conversational improvement-related question labeled as **Questions**. These questions should not explicitly tell the AI (you) what rules or standards to follow or what the specific goal should be. Instead, offer a high-level instruction that may hint at the criteria without stating them directly. The aim is to allow for creativity and subtle alignment with the criteria.
2. **Second Role**: Act as an expert model for improving articles. Provide:  
   - The improved version of the selected content labeled as **After Improvement**, designed to answer the **Questions** on **Before Improvement** above.  
   - A detailed explanation of the changes made, using **references from the paper context** to help answer the question and demonstrate alignment with the context and the criteria, labeled as **Explanation**.

--- the paper content starts  
A Differentiable Neural Computer (DNC) is a memory-augmented neural computation model capable of learning to solve complex algorithmic tasks, from simple sorting algorithms, through graph problems, to text question answering. In previous works, it was always given a constant number of planning steps to complete its task.
In this work, we argue that the number of planning steps the model is allowed to take, which we call "planning budget", is a constraint that can cause the model to generalize poorly and hurt its ability to fully utilize its external memory.
By introducing an adaptive planning budget that scales with input size during training, the model is better able to utilize its memory space, and achieves substantially better accuracy on input sizes not seen during training.
We experiment with Graph Shortest Path search, which has been used as a benchmark to measure these models in the past, and with the Graph MinCut problem.
In both problems, our proposed approach improves performance and generalizes better compared to the standard planning budget.

## 1 Introduction

In the past few years, Deep Neural Networks (DNNs) produced substantial advances in a myriad of fields, from image classification and segmentation, through audio analysis and speech recognition, to generation of art and text with human-like accuracy. Despite these advances, a significant problem still remains: generalization to unseen inputs. Since a DNN is trained by solving an optimization task over a training set, the network often achieves lower performance on inputs that are out of distribution to its training set. This can be attributed to sparse input distributions, outliers, edge cases and more. In order to improve generalization, DNNs are trained on ever increasing numbers of training samples. For example in NLP, dataset sizes can reach billions and trillions of tokens Kudugunta et al. (2023); Wang et al. (2023).

A potential solution to the issue of generalization lies in algorithms, which usually are able to solve a problem for all cases. Instead of learning to approximate a function, we learn to produce an algorithm; a series of steps, changing internal state, memory, or external interfaces, that eventually reach the desired outcome. We can hope that if a good algorithm is found, it will generalize to all cases by definition. This idea is often called Algorithmic Reasoning, and we use this name in this paper.

There are multiple examples of algorithmic reasoning, which can be implemented in an explicit or an implicit manner. In the explicit approach, the model's task is to output a description of the algorithm it has learned. Examples include AlphaTensor Fawzi et al. (2022), in which the model learns to find general matrix multiplication algorithms for various matrix sizes; code generation models such as Li et al. (2022), and Large Language Model (LLM) that are able to generate a piece of code solving a task described in free text Shinn et al. (2023).

In the implicit approach, the processor learns to output actions that work for a problem instance. For example in graph shortest path, the input is a specific graph \(G\) and a query pair of source node and target node \((s,t)\). The correct output is the edges comprising the shortest path between \(s\) and \(t\). The processor's internal representation mimics a computer model, and its actions can be interpreted as instructions for processing memory or register values. However unlike the explicit approach, which outputs instructions that should work for all inputs, the implicit approach simply runs the learned algorithm on the given inputs. In order to run the algorithm, we must run the model. In this way, the model learns to perform the algorithm rather than describe it; the model's weights, internal representation space, and architecture together comprise the learned algorithm. Examples include Zaremba and Sutskever (2016); Velickovic et al. (2020); Kurach et al. (2016); Graves et al. (2014)

An important example of this approach is the Differentiable Neural Computer model Graves et al. (2016), which is the focus of this work. In brief, the DNC is a Recurrent Neural Network (RNN) based on a differentiable implementation of a Turing Machine, extending previous work Graves et al. (2014). Featuring an LSTM with a memory matrix, the DNC can model algorithms that interact with external memory, handling tasks like copying, sorting, and graph problems. A summary of the DNC architecture is provided in Appendix G.

In short, The DNC processes inputs by iteratively absorbing a sequence of vectors, storing them in memory, and executing memory operations for task-specific outputs. It has several addressing mechanisms which allow it to allocate new cells, update existing ones orlookup a specific cell by its content. Its operation spans three phases: input, planning, and answering. Initially, it receives input, then undergoes \(p\) planning steps for processing--a number previously limited to zero or just 10 in more complex tasks--and finally produces the output in the answering phase.

We state and provide empirical evidence for two important claims:

1. Choosing an adaptive planning budget allows the model to learn a more general algorithm than it can with a constant budget, allowing it to generalize to input sizes much larger than seen during training.
2. Choosing an adaptive planning budget allows the model to learn better memory utilization, facilitating the use of larger memory for larger inputs.

Our findings show that a DNC trained with a constant \(p\) faces limitations, most likely overfitting to heuristics and average case-attributes of the input. This issue persists unless it is trained with a substantially large \(p\), which also has inherent limitations. In contrast, the adaptive model avoids these issues, demonstrating more robust performance without the constraints observed in the constant budget models. The paper is structured as follows: Section 2 overviews related work; Section 3 details our method and its complexity theory motivation; Section 4 presents experimental analysis supporting our first claim; Section 5 discusses larger constant budgets and evidence for our second claim; and Section 6 concludes the paper.
--- the paper content ends  

### Response Format (must be strictly followed):

--- Before Improvement starts  
[Selected content]  
--- Before Improvement ends  

--- Questions start  
[Concise, improvement-related question based on the criteria 'Clarity and Impact of Key Innovations and Findings' ]  
--- Questions end  

--- After Improvement starts  
[Revised version of the selected content to answer the **Questions** above]  
--- After Improvement ends  

--- Explanation starts  
[An explanation of the changes made, showing how they align with the context of the article and address the criteria. Include references from the paper context where relevant.]  
--- Explanation ends  
"""

# 调用 OpenAI GPT-4 模型 (使用新的 API 方式)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个智能的程序员。"},
        {"role": "user", "content": prompt}
    ],
    max_tokens=4000,
    temperature=0.7,
)

# 获取生成的回答 (新的响应格式)
generated_text = response.choices[0].message.content

# 打印生成的回答
print(generated_text)
