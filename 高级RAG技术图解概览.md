# 高级 RAG 技术：图解概览
>原文：[Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)
>作者：IVAN ILIN

本文全面研究了高级检索增强式生成技术 (RAG) 及其算法，系统地整理了各种方法。文章中还包含了我知识库中与提到的各种实现和研究相关的链接集。

## 引言

检索增强生成（Retrieval Augmented Generation，简称 RAG）为大语言模型 (LLMs) 提供了从数据源检索的信息，以此为基础生成回答。简而言之，RAG 结合了搜索技术和大语言模型的提示功能，即模型根据搜索算法找到的信息作为上下文来回答查询问题。无论是查询还是检索的上下文，都会被整合到发给大语言模型的提示中。<br>
2023 年，基于 RAG 架构的大语言模型系统成为最受欢迎的技术。许多产品几乎全依赖 RAG 架构，这包括结合网络搜索引擎和大语言模型的问答服务，以及数以百计的数据交互应用程序。<br>
即使是 **向量搜索** 领域也因 RAG 的热潮而得到推动，尽管基于嵌入式搜索引擎的技术早在 2019 年就已经运用了 [faiss](https://faiss.ai/)。例如 [chroma](https://github.com/chroma-core/chroma)、[weaviate.io](https://weaviate.io/) 和 [pinecone](https://www.pinecone.io/) 这样的向量数据库初创公司在现有的开源搜索索引基础上 — 主要是 faiss 和 [nmslib](https://github.com/nmslib/nmslib) — 构建了自己的系统，并在近期增加了输入文本的额外存储和其他工具。<br>
**两个最显著的开源库用于基于大语言模型的管道和应用程序** 是 [LangChain](https://python.langchain.com/docs/get_started/introduction) 和 [LlamaIndex](https://docs.llamaindex.ai/en/stable/)，它们分别在 2022 年 10 月和 11 月成立，相差一个月，受到 ChatGPT 发布的启发，并在 2023 年获得了广泛应用。

> 本文旨在系统地介绍高级 RAG 技术及其在 LlamaIndex 等平台的实现，以帮助开发者深入理解这项技术。
> 
> 问题在于，大多数教程仅挑选个别技术进行详细讲解，而没有全面介绍所有可用的工具。
> 
> 另一个问题是，LlamaIndex 和 LangChain 都是发展迅速的优秀开源项目，以至于它们的文档已经比 2016 年的机器学习教科书还要厚。

##  原始 RAG 方法

本文中所讨论的 RAG 流程起始于一系列文本文档——本文不涉及此之前的内容，而是将其交由能够连接到从 Youtube 到 Notion 等任何想象得到的来源的优秀开源数据加载器处理。

![作者设计的一个方案，以及文中接下来的所有方案](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_Ko_ihY8ecAukf2g1.webp)

作者设计的一个方案，以及文中接下来的所有方案<br>
**标准 RAG 方法** 简而言之，是这样的：你将文本分割成小块，然后使用某种 Transformer Encoder 模型将这些小块转换为向量，把这些向量汇总到一个索引中，最后创建一个针对大语言模型的提示，指导模型根据我们在搜索步骤中找到的上下文回答用户的查询。  
在实际运行中，我们用相同的 Encoder 模型将用户的查询转化为向量，然后对这个查询向量进行搜索，与索引进行匹配，找出最相关的前 k 个结果，从我们的数据库中提取相应的文本块，并将其作为上下文输入 LLM 进行处理。<br>
这样的提示可以是：
```
def question_answering(context, query):
    prompt = f"""
Give the answer to the user query delimited by triple backticks ```{query}```\
using the information given in context delimited by triple backticks ```{context}```.\
If there is no relevant information in the provided context, try to answer yourself,
but tell user that you did not have any relevant context to base your answer on.
Be concise and output the answer of size less than 80 tokens.
"""

    response = get_completion(instruction, prompt, model="gpt-3.5-turbo")
    answer = response.choices[0].message["content"]
    return answer
```

一个 RAG 提示的例子<br>
[**提示优化**](https://docs.llamaindex.ai/en/latest/examples/prompts/prompts_rag.html) 是提升 RAG 流程性能的一种简便有效的方法。不要忘了查阅 OpenAI 提供的详尽的 [提示优化指南](https://baoyu.io/translations/openai/openai-prompt-engineering-guides)。<br>
虽然 OpenAI 是大语言模型提供商的市场领头羊，但还有其他不少选择，例如 Anthropic 的 [Claude](https://www.anthropic.com/product)，Mistral 推出的最新潮流的小型但功能强大的模型 [Mixtral](https://mistral.ai/news/mixtral-of-experts/)，Microsoft 的 [Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)，以及如 [Llama2](https://huggingface.co/blog/llama2)，[OpenLLaMA](https://huggingface.co/openlm-research)，[Falcon](https://huggingface.co/tiiuae) 等众多开源选择，供你为 RAG 流程选用不同的“大脑”。

## 高级 RAG 技术概览

接下来，我们将详细探讨高级 RAG 技术。<br>  
下面这幅图展示了涉及的核心步骤和算法。 <br> 
为了使图解清晰易懂，我们省略了一些逻辑循环和复杂的多步骤智能体行为。

![这是高级 RAG 架构中的一些关键组成部分，更像是一套可选工具的集合，而不是固定的设计蓝图。](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_Gr_JqzdpHu7enWG9.webp)

这是高级 RAG 架构中的一些关键组成部分，更像是一套可选工具的集合，而不是固定的设计蓝图。

图中的绿色元素代表我们将进一步探讨的核心 RAG 技术，蓝色部分则是相关文本。并不是所有高级 RAG 概念都能在一张图上展示得一目了然，例如，我们省略了一些扩展上下文的方法 — 我们会在后续的讨论中逐步深入这些话题。

### 1 分块 (Chunking) & 向量化 (Vectorisation)

首先，我们需要为文档内容创建一个向量索引，然后在实时运行中，寻找这些向量与查询向量之间最小的余弦距离，这样可以找到与查询最接近的语义含义。

#### 1.1 分块 (Chunking)

Transformer 模型的输入序列长度是固定的。即便输入的上下文窗口很大，相比于几页文本的平均向量，一句话或几句话的向量更能准确地代表其语义含义（这也取决于具体的模型，但一般都是这样）。因此，你需要**将数据进行分块** —— 把初始文档分成一些块，每块大小适中，既不丢失原有的含义（比如，可以按句子或段落分割文本，而不是将单个句子切成两半）。有许多文本分割工具能够完成这个任务。<br>
**块的大小是一个需要深思熟虑的参数** —— 它取决于你所使用的嵌入模型以及它在 tokens 方面的容量。例如，标准的 Transformer 编码模型，如基于 BERT 的句子 Transformers 最多可以处理 512 个 tokens，而 OpenAI ada-002 能够处理更长的序列，例如 8191 个 tokens，但**这里需要权衡的是，为 LLM 提供足够的上下文进行推理，同时确保文本嵌入足够具体，以便有效地执行搜索**。在[这里](https://www.pinecone.io/learn/chunking-strategies/)，你可以找到一项研究，展示了选择块大小时的各种考虑因素。在 LlamaIndex 中，这一问题通过 [NodeParser 类](https://docs.llamaindex.ai/en/stable/api_reference/service_context/node_parser.html) 得到解决，它提供了定义自己的文本分割器、元数据、节点/块关系等高级选项。

#### 1.2 向量化 (Vectorisation)

下一步是选择一个 **模型来嵌入我们的块** —— 这里有很多选择。我个人倾向于使用 **为搜索优化的模型**，如 [bge-large](https://huggingface.co/BAAI/bge-large-en-v1.5) 或 [E5](https://huggingface.co/intfloat/multilingual-e5-large) 嵌入模型。你可以查看 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 了解最新的模型更新情况。<br>
想了解如何在端到端系统中实现块处理和向量化步骤吗？不妨参考 LlamaIndex 中完整数据摄取管道的[示例](https://docs.llamaindex.ai/en/latest/module_guides/loading/ingestion_pipeline/root.html#)。

### 2 搜索索引

#### 2.1 向量存储索引

![在这个架构及后续内容中，为了简化描述，我们不考虑编码器部分，直接把查询内容送入索引。显然，查询内容会首先进行向量化。类似地，尽管索引是根据向量而不是具体的块来进行检索的，但最终我们还是以块的形式展现结果，因为获取这些块相对简单。](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_fCxtcFf8gIgnaJfE.webp)<br>
在这个架构及后续内容中，为了简化描述，我们不考虑编码器部分，直接把查询内容送入索引。显然，查询内容会首先进行向量化。类似地，尽管索引是根据向量而不是具体的块来进行检索的，但最终我们还是以块的形式展现结果，因为获取这些块相对简单。<br>
**RAG 管道的核心是搜索索引**，它存储了我们在上一步骤中向量化的内容。最基本的实现方法是采用平面索引，即直接计算查询向量与所有块向量之间的距离。<br>
**为了实现高效检索，一个更高级的搜索索引**应该采用**向量索引**，比如 [faiss](https://faiss.ai/)、[nmslib](https://github.com/nmslib/nmslib) 或 [annoy](https://github.com/spotify/annoy)。这些工具基于近似最近邻居算法，如聚类、树结构或 [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) 算法。<br>
此外，还有一些托管服务，如 OpenSearch、ElasticSearch 和向量数据库，它们自动处理前文提到的数据摄取流程，例如 [Pinecone](https://www.pinecone.io/)、[Weaviate](https://weaviate.io/) 和 [Chroma](https://www.trychroma.com/)。<br>
根据您的索引选择、数据和搜索需求，**您还可以存储元数据**，并使用**元数据过滤器**来按照日期或来源等条件进行信息检索。<br>
LlamaIndex 支持多种[向量存储索引](https://docs.llamaindex.ai/en/latest/community/integrations/vector_stores.html)，同时也兼容其他简单的索引类型，如列表索引、树索引和关键词表索引。关于这些索引，我们会在后续的融合检索部分详细介绍。

#### 2.2 分层索引

![](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_nDwj0Jgpyk2qc_qJ.webp)

当需要从众多文档中检索信息时，高效的搜索、发现相关内容并整合成含有参考来源的答案变得尤为重要。在处理大型数据库时，一个高效的策略是**构建两个索引：一个包含摘要，另一个包含文档的各个部分**。这样的搜索分为两步：首先利用摘要来筛选出相关文档，然后只在这个筛选出的相关文档集中继续深入搜索。

#### 2.3 假设性问题和 HyDE

另一种策略是让大语言模型**为文档的每个部分产生一个问题，并把这些问题转换成数学上的向量**。在实际操作中，这些问题向量构成一个索引，用于对用户的查询进行匹配搜索（这里是用问题向量而非原文档的内容向量来构成索引），检索到相应问题后，再链接回原始文档的相应部分，作为大语言模型提供答案的背景信息。 <br>
这种方法通过增强查询与假设问题之间的语义相似性，从而提升了搜索的精准度，相比直接使用文档内容的方法效果更佳。<br>
此外，还有一种反向逻辑的方法，名为[**HyDE**](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf)。这种方法中，你会让大语言模型针对一个查询生成一个假设性的回应，然后结合这个回应的向量和查询的向量，共同用于提升搜索的效果。

#### 2.4 语境增强

**本节的核心理念是通过检索更小的信息块来提高搜索质量**，**同时为大语言模型增加更多周围语境以便其进行推理**。  <br>
我们有两种方法：一是通过增加检索到的小块周围的句子来扩大语境；二是递归地将文档分割成包含小块的大块。

[**2.4.1 句子窗口检索法**](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html)

在这种方法中，文档的每个句子都被单独编码，这样可以极大提高查询与语境之间的余弦距离搜索的准确性。  <br>
为了更好地分析找到的语境，我们在检索到的最相关单句之前后各扩展 _k_ 个句子，然后把这个扩展后的语境送给 LLM 进行推理。

![图中绿色部分代表在索引搜索中找到的句子编码，而整个黑色和绿色的段落则是提供给 LLM 以扩大其推理语境的内容。](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_JKZ9m_c6jyIKqCWu.webp)

图中绿色部分代表在索引搜索中找到的句子编码，而整个黑色和绿色的段落则是提供给 LLM 以扩大其推理语境的内容。

**2.4.2** [**自动合并检索法**](https://docs.llamaindex.ai/en/latest/examples/retrievers/auto_merging_retriever.html) **（也称为** [**父文档检索法**](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)**)**

这里的思路与句子窗口检索法类似 —— 搜索更精细的信息片段，然后在将这些内容送给 LLM 进行推理前扩大语境窗口。文档被分割成小的子块，这些子块又与更大的父块相关联。

![文档被分割成层次化的块结构，最小的叶子块被送至索引。在检索时，我们会找出 k 个叶子块，如果存在 n 个块都指向同一父块，我们就用这个父块替换它们，并把它送给 LLM 用于生成答案。](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_x4rMd50GP99OSDuo.webp)

文档被分割成层次化的块结构，最小的叶子块被送至索引。在检索时，我们会找出 k 个叶子块，如果存在 n 个块都指向同一父块，我们就用这个父块替换它们，并把它送给 LLM 用于生成答案。<br>
首先检索较小的数据块，如果在最初检索到的前 _k_ 个数据块中有超过 _n_ 个数据块与同一个父节点（即更大的数据块）相连，那么我们会用这个父节点来替换提供给大语言模型的上下文。这个过程类似于自动将几个检索到的小数据块合并成一个较大的父数据块，因此得名。值得一提的是，这种搜索只在子节点的索引中进行。想要深入了解，可以查阅 LlamaIndex 的教程，了解 [递归检索器 + 节点引用](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes.html)。

#### 2.5 融合检索或混合搜索

这是一个早已有之的想法：**结合传统的基于关键词的搜索和现代的语义或向量搜索**。传统搜索使用如 [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 或行业标准的 [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) 等稀疏检索算法，而现代搜索则采用语义或向量方法。这两种方法的结合就能产生出色的检索结果。  <br>
其关键在于如何恰当地融合这两种不同相似度得分的检索结果。这个问题通常通过 [互惠排名融合 (Reciprocal Rank Fusion)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 算法来解决，该算法能有效地对检索结果进行重新排序，以得到最终的输出结果。

![](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_0pQbhBEez7U-2knd.webp)

在 LangChain 中，这种方法是通过 [合奏检索器 (Ensemble Retriever)](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) 来实现的，该类将您定义的多个检索器结合起来，比如一个基于 faiss 的向量索引和一个基于 BM25 的检索器，并利用 RRF 算法进行结果的重新排序。<br>
在 LlamaIndex 中，这一过程也是以类似的方式 [实现](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html) 的。<br>
混合或融合搜索通常能提供更优秀的检索结果，因为它结合了两种互补的搜索算法——既考虑了查询和存储文档之间的语义相似性，也考虑了关键词匹配。

### 3 重新排名与过滤

在应用了前述的检索算法后，我们得到了初步的搜索结果。下一步是通过过滤、重新排列或转换这些结果来进一步优化它们。在 LlamaIndex 中，你可以找到多种[**后处理器**](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html)，这些处理器能够基于相似度分数、关键词、元数据进行过滤，或者使用其他模型如大语言模型、[句子 - 转换器交叉编码器](https://www.sbert.net/examples/applications/cross-encoder/README.html)和 Cohere 的重新排名 [接口](https://txt.cohere.com/rerank/)来重新排列结果，甚至可以根据元数据的日期新近度来操作——基本上包括了你能想到的所有功能。<br>
这一步是在将我们检索到的上下文送入大语言模型，以获得最终回答之前的最后环节。<br>
现在，我们将深入探索更为高级的 RAG 技术，比如查询转换和路由。这些技术涉及到大语言模型的使用，代表了一种更复杂的逻辑思维——在 RAG 流程中融合了 LLM 的推理能力。

### 4 查询变换

**查询变换是利用大语言模型作为推理引擎，对用户输入进行调整的一系列技术，目的是提升检索的质量。** 有多种方法可以实现这一点。

![查询变换的原理示意图](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_DP6RrSA2OkcHnWIV.webp)

查询变换的原理示意图

**对于复杂的查询，大语言模型能够将其拆分为多个子查询。** 比如，当你问：  
\_— “在 Github 上，Langchain 和 LlamaIndex 这两个框架哪个更受欢迎？”，  
\_我们不太可能直接在语料库的文本中找到它们的比较，所以将这个问题分解为两个更简单、具体的子查询是合理的：  
\_— “Langchain 在 Github 上有多少星？”  
— “Llamaindex 在 Github 上有多少星？”  
\_这些子查询会同时进行，检索到的信息随后被汇总到一个语句中，供大语言模型综合出对原始查询的最终答案。这两个功能分别在 Langchain 中以[多查询检索器](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever?ref=blog.langchain.dev)的形式和在 Llamaindex 中以[子问题查询引擎](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html)的形式实现。

1.  [**回溯提示**](https://arxiv.org/pdf/2310.06117.pdf?ref=blog.langchain.dev) 是指利用大语言模型来生成更广泛的查询，以此检索到更一般性或高层次的上下文，帮助我们更好地回答原始查询。 我们不仅对原始查询进行了检索，还将这两种上下文都输入到大语言模型中，以生成最终的答案。 这是 LangChain 的一个[示例实现](https://github.com/langchain-ai/langchain/blob/master/cookbook/stepback-qa.ipynb?ref=blog.langchain.dev)。
2.  **查询重写是指使用大语言模型来重新构造初始查询**，从而提高检索的效果。LangChain 和 LlamaIndex 都提供了这样的功能，虽然它们的方法有所不同，但我认为 LlamaIndex 在这方面的解决方案更为强大。

### 参考引用

这部分没有单独编号，因为它更多地被视为一种工具，而非一个改进检索的技术，但它同样重要。 **如果我们在生成答案时引用了多个不同的来源**，不管是因为初步查询的复杂性（需要执行多个子查询并合并结果），还是因为我们在不同文档中找到了针对单一查询的相关上下文，我们就会面临一个问题：如何能够**准确地标注我们所引用的各个来源**。<br>
实现这一点有几种方法：

1.  **在我们的提示中加入引用任务**，请求大语言模型明确提及所用来源的 id。
2.  **将生成的答案的部分内容与我们索引中的原始文本进行匹配** - LlamaIndex 提供了一种高效的[基于模糊匹配的方案](https://github.com/run-llama/llama-hub/tree/main/llama_hub/llama_packs/fuzzy_citation)。如果你不熟悉模糊匹配，这其实是一种[非常强大的字符串匹配技术](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536)。

### 5  [聊天引擎](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/root.html)

构建一个能对同一查询多次响应的优秀 RAG（检索增强生成）系统的关键之一是 **聊天逻辑，特别是要考虑到对话的上下文**，这一点与 LLM（大语言模型）时代之前的传统聊天机器人类似。  <br>
这是为了支持对后续问题、代词指代或与先前对话上下文相关的用户指令的处理。这个问题通过 **考虑聊天上下文的查询压缩技术** 来解决，并结合了用户的查询。<br>
处理这种上下文压缩有几种方法 —  
一个受欢迎且相对简单的方法是使用 [ContextChatEngine](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context.html)，它首先检索与用户查询相关的上下文，然后将其与来自 _内存_ 缓冲区的聊天历史一起发送给大语言模型，让大语言模型在生成下一步答案时能够了解之前的上下文。<br>
更为复杂的情况是 [CondensePlusContextMode](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context.html) — 在这种模式下，每次互动时都将聊天历史和最新消息压缩成一个新的查询，然后这个查询被发送到索引中，并将检索到的上下文与原始用户消息一起传递给大语言模型，以生成答案。<br>
值得一提的是，LlamaIndex 还支持基于 [OpenAI 智能体的聊天引擎](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_openai.html)，提供了一种更灵活的聊天模式，而 Langchain 也 [支持](https://python.langchain.com/docs/modules/agents/agent_types/openai_multi_functions_agent) OpenAI 的功能性 API。

![不同聊天引擎类型和它们的工作原理示意图](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_9cxhMMkUf8veRnRB.webp)
<center>不同聊天引擎类型和它们的工作原理示意图</center>

还有像 [ReAct 智能体](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_react.html) 这样的其他聊天引擎类型，但我们接下来将直接跳转到第 7 节，讨论智能体本身。

### 6 查询路由

**查询路由是指在接收到用户的查询后，由大语言模型决定接下来的操作步骤** - 常见的做法包括概述查询内容、对特定数据索引进行搜索，或尝试多个不同的处理方法，并将这些方法的结果合成一个答案。<br>
查询路由器的另一个作用是选择数据存储位置来处理用户查询。这些数据存储位置可能是多样的，比如传统的向量存储、图形数据库或关系型数据库，或者是不同层级的索引系统。在处理多文档存储时，通常会用到摘要索引和文档块向量索引这两种不同的索引。<br>
**设置查询路由器包括决定它可以做出哪些选择。**<br>
选择特定路由的过程是通过大语言模型来实现的，其结果按照预定义的格式返回，以指导查询到达指定的索引。如果是涉及到关联操作，这些查询还可能被发送到子链或其他智能体，如下面的**多文档智能体方案**所展示的那样。<br>
[LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/root.html) 和 [LangChain](https://python.langchain.com/docs/expression_language/how_to/routing?ref=blog.langchain.dev) 都提供了对查询路由器的支持。

### 7 RAG 中的 AI 智能体

AI 智能体（得到了 [Langchain](https://python.langchain.com/docs/modules/agents/) 和 [LlamaIndex](https://docs.llamaindex.ai/en/latest/use_cases/agents.html#) 的支持）几乎从第一个大语言模型 API 发布之初就已经出现了——**这个构想旨在赋予一个具备推理能力的大语言模型，配备各种工具和一个特定的任务**。这些工具可能包括诸如代码功能、外部 API 或其他智能体等确定性功能——LangChain 正是由这种大语言模型链接的理念得名。<br>
智能体本身非常复杂，而且不可能在一个 RAG 概览文章中对它进行深入探讨。因此，我将继续介绍基于智能体的多文档检索案例，并简要提及 OpenAI 助手，因为它是一个相对较新的概念，在[最近的 OpenAI 开发者大会上作为 GPTs 呈现](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)，并在下文将要介绍的 RAG 系统中发挥作用。<br>
[**OpenAI 助手**](https://platform.openai.com/docs/assistants/overview) 基本上整合了我们以前在开源领域看到的许多围绕大语言模型的工具——如聊天历史、知识存储、文档上传界面，尤其是[**函数调用 API**](https://platform.openai.com/docs/assistants/tools/function-calling)。这一功能可以将自然语言指令转换为对外部工具或数据库的 API 调用。<br>
在 LlamaIndex 中，[OpenAIAgent](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent.html) 类将这种先进的逻辑与 ChatEngine 和 QueryEngine 类结合在一起，实现了基于知识和上下文的智能聊天，以及在单次对话中调用多个 OpenAI 功能的能力，从而真正实现了智能的代理行为。<br>
让我们来看一下[**多文档智能体**](https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents.html)的**方案** —— 这是一个颇为复杂的配置，涉及到在每个文档上初始化一个智能体（[OpenAIAgent](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent.html)），该智能体能进行文档摘要制作和传统问答机制的操作，**还有一个顶层智能体**，负责将查询分配到各个文档智能体，并综合形成最终的答案。<br>
每个文档智能体配备了两种工具 —— 向量存储索引和摘要索引。它会根据被分配的查询来决定使用哪一个工具。 对顶层智能体而言，所有的文档智能体都是其工具。<br>
这个方案展示了一种高级 RAG 架构，其中每个参与的智能体都需要做出众多的路由决策。**这种方法的优势在于，它可以对描述于不同文档及其摘要中的不同解决方案或实体进行比较**，同时还包括了传统的单文档摘要制作和问答机制 —— 这大致涵盖了与文档集合交互的常见用例。

![图示展示了涉及查询路由和智能体行为模式的多文档智能体方案。](https://baoyu.io/images/rag/advanced-rag-techniques-an-illustrated-overview/0_FZp2J2NyHHBXPtii.webp)

图示展示了涉及查询路由和智能体行为模式的多文档智能体方案。<br>
这种复杂配置的缺点可以通过图片理解 —— 由于需要在智能体内部的大语言模型之间进行多次往返迭代，其运行速度较慢。顺便一提，LLM 调用通常是 RAG 管道中耗时最长的操作，而搜索则是出于设计考虑而优化了速度。因此，对于大型的多文档存储，我建议考虑对此方案进行简化，以便实现扩展。

### 8 响应合成阶段

这是任何 RAG 流程的最终步骤 —— 基于我们精心检索的所有上下文和用户的初始查询来生成答案。最简单的方式是将所有高相关性的检索上下文和查询一起直接输入到一个大语言模型 (大语言模型) 中。然而，还有其他更为复杂的方法，例如通过多次调用大语言模型来精细化检索的上下文，从而得出更优的答案。<br>
\*\*响应合成的主要方法包括：  
\*\*1. 通过逐块发送检索的上下文到大语言模型，以**逐步优化答案**2\. **概括检索的上下文**，使其适应提示条件 3. 根据不同的上下文块**生成多个答案**，然后将它们**整合或概括**。更多详情请参阅[响应合成器模块文档](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html)。

## 编码器与大语言模型的微调

这个方法涉及对 RAG 流程中的两个深度学习模型之一进行微调 —— 或是负责嵌入质量和上下文检索质量的 Transformer **编码器**，或是负责最有效利用提供上下文来回答用户查询的 **大语言模型** —— 幸运的是，后者擅长于少样本学习。<br>
如今，我们能够利用像 GPT-4 这样的高端大语言模型来创建高质量的合成数据集，这是一个巨大的优势。但是，我们也应当注意到，仅用小型合成数据集对专业研究团队开发的、基于大量精心收集、清洁和验证的数据集训练的开源模型进行快速调整，可能会降低模型的整体性能和能力。

### 编码器微调 (Encoder fine-tuning)

我曾对使用编码器微调来增强 Transformer 编码器的效率持有一些怀疑，毕竟最新的搜索优化 Transformer 编码器已经相当高效。然而，在 [LlamaIndex 笔记本](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding.html) 中，我测试了对 [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)（在撰文时位于 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 前四）的微调效果。结果显示，这种微调使检索质量提升了 2%。虽然这个提升不是特别显著，但了解这种选择对于特定领域数据集的 RAG (Retrieval-Augmented Generation，检索增强生成) 构建来说是非常有价值的。

### 排序器微调 (Ranker fine-tuning)

**另一个值得考虑的老方法是，在你不完全信任基础编码器的情况下，使用交叉编码器 (cross-encoder) 对检索到的结果进行重新排列。** 这个过程是这样的：你把查询和每个前 k 个检索到的文本块一起送入交叉编码器，中间用 SEP (分隔符) Token 分隔，并对它进行微调，使其对相关的文本块输出 1，对不相关的输出 0。一个这种微调过程的优秀例子可以在[这里](https://docs.llamaindex.ai/en/latest/examples/finetuning/cross_encoder_finetuning/cross_encoder_finetuning.html#)找到，结果显示通过交叉编码器微调，成对比较得分提高了 4%。

### 大语言模型微调

OpenAI 最近推出了大语言模型微调的 [API](https://platform.openai.com/docs/guides/fine-tuning) ，同时 LlamaIndex 也提供了一个教程，教你如何在 RAG（检索式问答生成）环境中对 GPT-3.5-turbo 进行微调，以便从 GPT-4 中获取知识。这个方法的核心思想是：先用 GPT-3.5-turbo 从一个文档中生成一系列问题，然后利用 GPT-4 根据文档内容来回答这些问题，通过这种方式构建基于 GPT-4 的 RAG 处理流程，接着再对这批问题和答案对进行微调。使用的 [ragas](https://docs.ragas.io/en/latest/index.html) 框架在 RAG 流程评估中表现出了显著的提升，微调后的 GPT 3.5-turbo 在生成答案时，相比原始模型更好地利用了所给的上下文，忠实度提高了 5%。<br>
Meta AI 研究团队最近的一篇 [论文](https://arxiv.org/pdf/2310.01352.pdf) 中介绍了一种更为复杂的方法，名为 RA-DIT（检索增强的双向指令调整），提出了一种同时对大语言模型和检索器（论文中称为双编码器）进行调整的技术。这种技术依据问题、上下文和答案的三元组来进行调整。更多实施细节请参考这个 [指南](https://docs.llamaindex.ai/en/stable/examples/finetuning/knowledge/finetune_retrieval_aug.html#fine-tuning-with-retrieval-augmentation)。这种技术不仅被用于通过微调 API 对 OpenAI 的大语言模型进行微调，也被用于 Llama2 的开源模型，相比于搭载 RAG 的 Llama2 65B，它在知识密集型任务上的表现提升了约 5%，在常识推理任务上也有所提升。<br>
_如果您对 RAG 环境下的大语言模型微调有更好的方法，请在评论区分享您的见解，尤其是那些适用于小型开源大语言模型的方法。_

## 评估

目前存在**多种框架用于评估 RAG (Retrieval-Augmented Generation) 系统的性能**，它们普遍采用了几项独立的评估指标，如**答案的相关性、答案的基于性、真实性和检索到的内容的相关性**。<br>

在之前章节提及的 [Ragas](https://docs.ragas.io/en/latest/index.html)，使用[真实性](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html)和[答案相关性](https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html)来评价生成答案的质量，并使用传统的上下文[精准度](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html)和[召回率](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html)来评估 RAG 方案的检索性能。<br>
在安德鲁 NG 最近推出的精彩短期课程[构建和评估高级 RAG](https://learn.deeplearning.ai/building-evaluating-advanced-rag/)中，以及其中提到的 LlamaIndex 和评估框架[Truelens](https://github.com/truera/trulens/tree/main)，他们提出了一个**RAG 三元组**评估模式 — 分别是对问题的**检索内容相关性**、答案的**基于性**（即大语言模型的答案在多大程度上得到了提供的上下文的支持）和答案对问题的**相关性**。<br>
其中最关键且可控的指标是**检索内容的相关性** — 实际上是上述高级 RAG 管道的前 1-7 部分加上编码器和排名器的微调部分，这些都是为了提高这个指标。而第 8 部分和大语言模型的微调则专注于提高答案的相关性和基于性。<br>
一个简单有效的检索器评估管道的例子可以在[这里](https://github.com/run-llama/finetune-embedding/blob/main/evaluate.ipynb)找到，它已被应用于编码器的微调部分。一个更高级的方法不仅考虑**命中率**，还包括了常用的搜索引擎评估指标**平均倒数排名 (Mean Reciprocal Rank)**，以及生成答案的质量指标，如真实性和相关性，这在 OpenAI 的[实用指南](https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Evaluate_RAG_with_LlamaIndex.ipynb)中有所展示。<br>
LangChain 提供了一个颇为先进的评估框架 [LangSmith](https://docs.smith.langchain.com/)。在这个框架中，你不仅可以实现自定义的评估器，还能监控 RAG 管道内的运行轨迹，进而增强系统的透明度。<br>
如果你正在使用 LlamaIndex 进行构建，可以尝试 [rag\_evaluator llama pack](https://github.com/run-llama/llama-hub/tree/dac193254456df699b4c73dd98cdbab3d1dc89b0/llama_hub/llama_packs/rag_evaluator)。这是一个便捷的工具，能够帮助你利用公共数据集来对你的管道进行评估。

## 结论

我试图勾勒出 RAG 的核心算法方法，并展示其中的一些，希望这能激发你在 RAG 流程中尝试一些新思路，或者为今年涌现的众多技术带来一定的系统性 — 对我而言，2023 年是迄今为止在机器学习领域最令人兴奋的一年。<br>
**还有许多其他因素需要考虑**，例如 **基于网络搜索的 RAG**（如 LlamaIndex 的 [RAGs](https://github.com/run-llama/rags)、[webLangChain](https://blog.langchain.dev/weblangchain/) 等），更深入地探索 **智能体架构**（以及最近 [OpenAI 在这个领域的投资](https://blog.langchain.dev/openais-bet-on-a-cognitive-architecture/)），以及一些关于 [**大语言模型长期记忆**](https://blog.langchain.dev/adding-long-term-memory-to-opengpts/) 的想法。<br>
**RAG 系统的主要生产挑战除了回答的相关性和忠实度外，还有速度**，尤其是如果你喜欢更灵活的基于智能体的方案，但这是另一篇文章的内容。ChatGPT 和大多数其他助手使用的这种流式特性不是随机的赛博朋克风格，而只是一种缩短感知答案生成时间的方法。 这就是为什么我看好小型大语言模型和最近的 Mixtral 和 Phi-2 发布，它们正在引导我们走向这个方向。<br>

>非常感谢你阅读这篇长文！

>主要参考资料收集在我的知识库中，有一个 co-pilot 可以与这组文档对话： [_https://app.iki.ai/playlist/236_](https://app.iki.ai/playlist/236)

>您可以在 [_LinkedIn_](https://www.linkedin.com/in/ivan-ilin-/) 或 [_Twitter_](https://twitter.com/ivanilin9) 上找到我