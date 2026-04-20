# FinGPT：开源金融大语言模型

**作者**: Hongyang Yang¹, Xiao-Yang Liu², Christina Dan Wang³∗

¹AI4Finance 基金会†；²哥伦比亚大学；³上海纽约大学

contact@ai4finance.org

---

## 摘要

大语言模型（LLMs）在各种领域展现了变革自然语言处理的潜力，引发了金融界的极大兴趣。然而，金融领域呈现出独特的挑战，包括高度的时间敏感性、持续的动态性以及低信噪比（SNR）。虽然像 BloombergGPT 这样的专有模型利用了其独特的数据积累，但这种特权访问需要一个开源替代方案来民主化互联网规模的金融数据。

在本文中，我们展示了面向金融部门的开源大语言模型 **FinGPT**。与专有模型不同，FinGPT 采用以数据为中心的方法，为研究人员和从业者提供可访问且透明的资源，以定制他们的金融大语言模型（FinLLMs）。我们强调了自动数据策划管道和轻量级低秩适应技术在构建 FinGPT 中的重要性。此外，我们提供基础任务作为基准测试的构建模块，并展示潜在应用作为用户的垫脚石，如智能投顾和情感分析。通过开源 AI4Finance 社区内的协作努力，FinGPT 旨在激发创新、民主化 FinLLMs，并在开放金融中解锁新机遇。

两个相关代码仓库：
- https://github.com/AI4Finance-Foundation/FinGPT
- https://github.com/AI4Finance-Foundation/FinNLP

---

## 1 引言

人工智能的持续扩展和演变为大语言模型（LLMs）的激增提供了肥沃的土壤 [Vaswani et al., 2017; Radford et al., 2018; Devlin et al., 2018; Ethayarajh, 2019; Lewis et al., 2019; Lewis et al., 2020; Brown et al., 2020; Thoppilan et al., 2022]，从而在不同领域引发了变革性转变。这一广泛变化引发了人们对金融大语言模型（FinLLMs）潜在应用的浓厚兴趣。然而，显而易见的是，获取高质量、相关且最新的数据是开发有效和高效 FinLLMs 的关键因素。

在金融领域利用大语言模型揭示了复杂的障碍。首先，存在**高度时间敏感性**问题。金融数据以其时间敏感特性为特征。一旦发布，能够推动市场的新闻或更新为投资者提供了一个狭窄的机会窗口来最大化他们的 alpha（投资相对回报的衡量标准）。其次，金融格局的特点是**高度动态性**。由于新闻、社交媒体更新和其他市场相关信息的不断流动，它处于持续变化状态。鉴于这些持续变化，频繁重新训练大语言模型不仅昂贵而且不切实际。最后，金融数据通常具有**低信噪比（SNR）** [Yang et al., 2020]。有用信息通常隐藏在大量不相关或嘈杂的数据中。从这片信息海洋中提取有价值的见解需要先进技术。

在专有领域，像 BloombergGPT [Wu et al., 2023] 这样的模型利用其对专有数据的独家访问来训练 FinLLM。然而，其数据收集和训练协议的可访问性和透明度受限，突显了对开放和包容性替代方案的需求。为了响应这一需求，我们正见证着开放金融领域中民主化互联网规模金融数据的趋势转变。

在本文中，我们解决了与金融数据相关的上述挑战，并介绍了 **FinGPT**，这是一个面向金融大语言模型（FinLLMs）的端到端开源框架。FinGPT 采用以数据为中心的方法，强调数据获取、清洗和预处理在开发开源 FinLLMs 中的关键作用。通过倡导数据可访问性，FinGPT 致力于加强金融领域的研究、协作和创新，为开放金融实践铺平道路。

**我们的贡献总结如下：**

- **以数据为中心的方法**：认识到数据策划的重要性，FinGPT 采用以数据为中心的方法，并实施严格的清洗和预处理方法来处理各种数据格式和类型。

- **端到端框架**：FinGPT 采用包含五层的全栈框架：
  - **数据源层**：确保全面的市场覆盖，通过实时信息捕捉解决金融数据的时间敏感性。
  - **数据工程层**：为实时 NLP 数据处理而优化，解决金融数据中固有的高时间敏感性和低信噪比挑战。
  - **大语言模型层**：专注于一系列微调方法，解决金融数据的高度动态性，确保模型的相关性和准确性。
  - **任务层**：负责执行基础任务。这些任务作为 FinLLMs 领域性能评估和交叉比较的基准。
  - **应用层**：展示实际应用和演示，突出 FinGPT 在金融领域的潜在能力。

- **民主化**：作为开源框架，FinGPT 旨在民主化金融数据和 FinLLMs，挖掘开放金融中未开发的潜力。我们将 FinGPT 视为激发金融领域创新的催化剂。FinGPT 不仅限于提供技术贡献，还培育 FinLLMs 的开源生态系统，促进实时处理和用户的定制化适应。通过在开源 AI4Finance 社区内培育强大的协作生态系统，FinGPT 有望完善我们对 FinLLMs 的理解和应用。

---

## 2 相关工作

### 2.1 FinLLMs 的崛起

大语言模型（LLMs）已被认为是自然语言处理领域的技术突破，如 GPT-3 和 GPT-4 [Brown et al., 2020; Jiang et al., 2023; OpenAI, 2023; Team et al., 2023; Liu et al., 2024]。它们采用基于 Transformer 的架构，在各种文本生成任务中展现出令人印象深刻的性能。作为 OpenAI 开发的 GPT 家族的分支，ChatGPT 旨在根据输入提示生成类似人类的文本。它已在各种应用中显示出显著效用，从起草电子邮件到编写代码，甚至创作艺术内容。

大语言模型已应用于金融领域内的各种任务 [Dredze et al., 2016; Araci, 2019; Bao et al., 2021; DeLucia et al., 2022]，从预测建模到从原始金融数据生成有洞察力的叙述。最近的文献专注于将这些模型用于金融文本分析，鉴于该领域丰富的文本数据，如新闻文章、财报电话会议记录和社交媒体帖子。

FinLLMs 的第一个例子是 BloombergGPT [Wu et al., 2023]，它在金融和通用数据源的混合数据集上训练。尽管其能力令人印象深刻，但存在访问限制，且高昂的训练成本激发了对低成本领域适应的需求。

我们的 FinGPT 响应上述挑战，提出开源 FinLLM。它采用人类反馈强化学习（RLHF）来理解和适应个人偏好，为个性化金融助手铺平道路。我们的目标是将 ChatGPT 等通用大语言模型的优势与金融适应相结合，利用大语言模型在开放金融中的能力。

### 2.2 为什么需要开源 FinLLMs？

AI4Finance 基金会 是一个非营利的开源组织，整合人工智能（AI）和金融应用。基金会拥有培育 FinTech 工具创新生态系统的良好记录，如 FinRL [Yang et al., 2020] 和 FinRobot [Yang et al., 2024]，正准备加速 FinLLMs 的演进。坚定的承诺和前沿贡献可能为 AI 在开放金融中的变革性应用铺平道路。

- **通过民主化 FinLLMs 推进平等机会**：采用开源方法促进对最先进技术的普遍访问，遵循民主化 FinLLMs 的理念。

- **培养透明度和信任**：开源 FinLLMs 提供其基础代码库的全面概述，增强透明度和信任。

- **加速研究和创新**：开源模式推动 AI 领域内的研究和开发进展。它允许研究人员利用现有模型，从而培育更快的创新和科学发现进程。

- **加强教育**：开源 FinLLMs 作为强大的教育工具，为学生通过直接参与完全可操作的模型来探索 FinLLMs 的复杂性提供前景。

- **通过社区协作升级金融文本数据的基础设施**：这种协作参与增强了模型的长期耐用性和有效性。

---

## 3 FinGPT 概述：FinLLMs 的开源框架

FinGPT 代表专门为 FinLLMs 设计的创新开源框架。如图 1 所示，FinGPT 由四个组件组成：数据源、数据工程、大语言模型和应用。每个组件在维持 FinGPT 的功能和适应性方面发挥着关键作用。

- **数据源层**：起点是数据源层，它协调从广泛的在线来源获取大量金融数据。该层通过整合来自新闻网站、社交媒体平台、财务报表、市场趋势等的数据，确保全面的市场覆盖。目标是捕捉市场的细微差别，从而解决固有的时间敏感性。

- **数据工程层**：该层专注于文本数据的实时处理，以解决金融数据中固有的高时间敏感性和低信噪比挑战。它融合了最先进的 NLP 技术来过滤噪声并突出最重要的信息片段。

- **大语言模型层**：位于核心位置，它包含各种微调方法，优先考虑轻量级适应，以保持模型的更新和相关性。通过保持模型的更新，FinGPT 可以应对金融数据的高度动态性，确保其响应与当前金融形势同步。

- **任务层**：任务层旨在提供构建模块。该层具有双重目的：首先，它执行 FinLLMs 领域 crucial 的各种基础任务，如情感分析、内容摘要和数值推理。其次，它建立标准化的指标和属性集。这些标准化元素不仅作为指标，还作为基准，促进 FinLLMs 领域内的性能评估和比较分析。

- **应用层**：FinGPT 的最后一个组件是应用层，旨在展示 FinGPT 的实际适用性。它为金融任务提供实践教程和演示应用，包括智能投顾服务和情感分析。这些实际演示不仅作为潜在用户的指南，还强调了 FinLLMs 的变革潜力。

### 3.1 数据源

FinGPT 的第一阶段涉及从广泛的在线来源收集大量金融数据。这些来源包括但不限于：

- **金融新闻**：路透社、CNBC、雅虎财经等网站是金融新闻和市场更新的丰富来源。这些网站提供有关市场趋势、公司收益、宏观经济指标和其他金融事件的宝贵信息。

- **社交媒体**：Twitter、Facebook、Reddit、微博等平台提供有关公众情绪、热门话题以及对金融新闻和事件的即时反应的丰富信息。

- **文件**：金融监管机构网站（如美国的 SEC）提供对公司文件的访问。这些文件包括年度报告、季度收益、内幕交易报告和其他重要的公司特定信息。证券交易所（纽约证券交易所、纳斯达克、上海证券交易所等）的官方网站提供有关股票价格、交易量、公司上市、历史数据和其他相关信息的关键数据。

- **趋势**：Seeking Alpha、Google Trends 以及其他金融博客和论坛等网站提供分析师意见、市场预测、特定证券或市场板块的走势以及投资建议。

- **学术数据集**：基于研究的数据集，为金融分析提供经过策划和验证的信息。

为了利用来自这些多样化来源的丰富信息，FinGPT 结合了数据获取工具，能够抓取结构化和非结构化数据，包括 API、网络爬虫工具以及在可用情况下的直接数据库访问。此外，系统设计尊重这些平台的服务条款，确保数据收集符合道德和法律要求。

**数据 API**：在 FinGPT 框架中，API 不仅用于初始数据收集，还用于实时数据更新，确保模型在最新数据上训练。此外，实施了错误处理和速率限制策略，以尊重 API 使用限制并避免数据流中断。

### 3.2 金融 NLP 的实时数据策划管道

金融市场实时运作，对新闻和情绪高度敏感。证券价格可能因新信息而迅速变化，处理该信息的延迟可能导致错过机会或增加风险。因此，实时处理在金融 NLP 中至关重要。

实时 NLP 管道的主要挑战是有效管理和处理持续流入的数据。管道中的第一步是设置一个系统来实时摄取数据。这些数据可能来自我们的数据源 API。以下是为数据摄取设计实时 NLP 管道的步骤。

**数据清洗**：实时数据可能嘈杂且不一致。因此，实时数据清洗涉及移除不相关数据、处理缺失值、文本归一化（如小写化）和错误纠正。

**分词**：在实时应用中，分词必须即时执行。这涉及将文本流分解成更小的单元或标记。

**向量嵌入**：FinGPT 使用领域适应的嵌入模型将策划的金融文本编码为密集语义向量。嵌入过程结合了实体感知表示（股票代码、比率、事件）和时间元数据，使系统能够捕捉细粒度的金融含义。所有嵌入都在向量数据库中索引，以支持低延迟检索，支持 RAG、事件聚类和与市场一致的 RLSP 训练。

**特征提取**：特征提取涉及将原始数据转换为 ML 模型可以理解的输入。在实时系统中，这通常需要是一个快速且高效的过程。可以使用 TF-IDF、词袋模型或 Word2Vec 等嵌入向量等技术。

**数据增强**：在金融市场的动态格局中，增强训练数据的多样性和数量对于构建稳健的 NLP 模型至关重要。将采用数据增强策略来生成可以模仿实际金融数据特征的综合数据。

### 3.3 大语言模型（LLMs）

一旦数据得到适当准备，就与大语言模型一起使用以生成有洞察力的金融分析。大语言模型层包括：

- **大语言模型 API**：已建立的大语言模型 API 提供基础语言能力，作为进一步模型开发和定制的基础。

- **可训练模型**：用户可以在私有数据上微调 FinGPT 的可训练模型，用于个性化金融应用，确保特定用例的相关性和准确性。

- **微调方法**：FinGPT 支持各种微调方法，促进其高效有效地适应个性化智能投顾。

- **提示工程**：提示工程对于优化大语言模型的输入查询、增强准确金融信息的提取至关重要。这个迭代过程需要精心制作提示以获得细微的响应，需要对金融和语言模型特性的深刻理解。

**为什么为金融轻量级微调大语言模型？**

正如 [Ouyang et al., 2022] 中所述，为金融微调或指令微调现有大语言模型，与从头开始重新训练模型的昂贵且漫长的过程相比，提供了一种节省成本和时间的替代方案。

BloombergGPT [Wu et al., 2023] 尽管在其金融特定能力方面表现出色，但计算需求密集。它使用了约 130 万 GPU 小时进行训练，按 AWS 云的每小时 2.3 美元费率计算，转化为每次训练约 300 万美元的惊人成本。与 BloombergGPT 等高计算成本模型相比，FinGPT 通过专注于顶级开源大语言模型的轻量级适应，提供了更易访问的解决方案。适应成本显著下降，估计每次微调约 300 美元。

这种方法确保了及时更新和适应性，这在动态金融领域至关重要。作为开源项目，FinGPT 不仅促进透明度，还允许用户定制，满足个性化金融咨询服务兴起的趋势。最终，FinGPT 具有成本效益、灵活的框架有潜力民主化金融语言建模并培育以用户为中心的金融服务。

**通过低秩适应（LoRA）微调**

在 FinGPT 中，我们利用金融数据集微调预训练的大语言模型。众所周知，高质量的标记数据是许多成功大语言模型（包括 ChatGPT）的关键决定因素。然而，获取这种顶级标记数据通常在时间和资源方面成本高昂，通常需要金融专业知识。

当设想将大语言模型应用于审查金融文本和促进量化交易策略时，必须考虑利用金融市场内可用的内在标记机制。鉴于此，FinGPT 采用与个别新闻文章相对应的股票价格变化百分比作为输出标签。通过分配预定阈值，这些连续标签被分类为三个离散情感类别：正面、负面和中性。

同时，在提示工程阶段，模型被精心指示从三个情感类别中选择一个作为其输出。这种细致的方法确保最大限度地利用预训练期间获得的信息，促进金融情感产生有洞察力和可靠的预测。对大语言模型实施低秩适应（LoRA）[Hu et al., 2021; Dettmers et al., 2023] 及其量化变体 QLoRA [Dettmers et al., 2023]，通过将可训练参数数量从压倒性的 61.7 亿减少到可管理的 367 万，显著简化了模型。

**通过股票价格强化学习（RLSP）微调**

同样，我们可以用股票价格强化学习（RLSP）替代 ChatGPT 使用的人类反馈强化学习。这种替代背后的推理是，股票价格提供了可量化的客观指标，反映市场对新闻和事件的反应。这使其成为训练我们模型的强大实时反馈机制。

强化学习（RL）允许模型通过与环境互动并接收反馈来学习。在 RLSP 的情况下，环境是股票市场，反馈以股票价格变化的形式出现。这种方法允许 FinGPT 完善其对金融文本的理解和解释，提高其预测各种金融市场事件市场反应的能力。

通过将新闻情感与相关股票的后续表现联系起来，RLSP 提供了一种有效的方法来微调 FinGPT。本质上，RLSP 允许模型推断市场对不同新闻事件的反应，并相应地调整其理解和预测。

因此，将 RLSP 整合到 FinGPT 的微调过程中，为提高了模型的金融市场理解和预测准确性提供了强大工具。通过使用实际股票价格变动作为反馈，我们直接利用市场智慧使我们的模型更有效。

**检索增强生成（RAG）**

检索增强生成（RAG）是 FinGPT 内融合的关键技术 [Zhang et al., 2023]，因为它无缝结合了上下文检索机制和大语言模型（LLMs）的能力来优化语言生成任务。这个细致的过程确保大语言模型不是在真空中生成内容，而是在其输出中信息丰富且细致，从检索文档提供的丰富上下文背景中汲取。这些文档与输入提示协同工作，有效地引导大语言模型制作不仅准确而且深深植根于相关上下文的响应，从而提高生成文本的实用性和可靠性。

### 3.4 基础任务

FinGPT 作为金融领域的多功能工具，通过有效过滤和分析信息，为专业人士和个人提供宝贵帮助。该模型在以下基础任务中表现出色：

- **摘要**：FinGPT 可以将冗长的金融文档高效地压缩成简洁的摘要，保留关键信息和见解。此功能对于快速理解综合报告、新闻文章或财务报表的精髓而无须浏览整个内容具有不可估量的价值。

- **命名实体识别（NER）**：该模型擅长识别和分类文本中的命名实体，如公司名称、股票代码、货币价值和百分比。此能力对于从非结构化文本中提取特定数据点至关重要，促进更结构化和信息丰富的分析。

- **信息提取**：FinGPT 可以从各种来源精心提取相关信息，为用户提供宝贵见解。此能力对于决策过程至关重要，因为它筛选噪声以突出基本数据和趋势。

- **情感分析**：情感分析作为基础任务至关重要，因为其在识别市场情绪（即金融情感分析）和在智能投顾平台中用于在产品推荐期间识别客户情绪的双重应用。

- **数据分析**：FinGPT 可以处理和分析庞大数据集，识别数据中的模式、异常和显著变化。此功能支持数据驱动的决策，提供对市场动态和财务绩效的更清晰理解。

- **数值推理**：该模型可以基于文本中提供的数据执行计算和数值分析，支持用户评估财务指标、进行预测和有效评估风险。

- **术语理解**：FinGPT 擅长理解和解释复杂的金融术语和行话，使其成为资深专业人士和金融领域新人的宝贵助手。

- **意图检测**：该模型可以准确识别查询背后的用户意图，促进更有效和相关的响应。此功能对于开发直观且用户友好的金融咨询应用和服务特别有用。

各种开源数据集作为基准，有效地参与多种基础任务。示例包括 BloombergGPT [Wu et al., 2023]，它利用从 FLUE 基准 [Shah et al., 2022] 衍生的精选金融数据集。这些数据集用于一系列基本任务，如情感分析和 NER。其他值得注意的数据集包括 FinRED [Sharma et al., 2022]（用于信息提取任务）、FINQA [Chen et al., 2021]（用于数值推理评估）和 FinRAD [Ghosh et al., 2021]（对于理解和识别金融术语至关重要）。

### 3.5 潜在应用

FinGPT 可在金融服务中找到广泛应用，作为强大的信息过滤器帮助专业人士和个人。潜在应用包括：

- **金融情感分析**：评估不同金融平台的情绪，以获得有洞察力的投资指导。

- **智能投顾**：FinLLMs 内的智能投顾功能在提供个性化财务建议方面发挥关键作用，最大限度地减少持续人工咨询的必要性。

- **量化交易**：生成交易信号以进行明智的交易决策。

- **投资组合优化**：利用众多经济指标和投资者档案进行最佳投资组合构建。

- **信用评分**：从金融数据预测信用度，帮助贷款决策。

- **并购（M&A）预测**：通过分析财务数据和公司档案预测潜在的并购活动，帮助投资者预测市场走势。

- **ESG（环境、社会、治理）评分**：通过分析公共报告和新闻文章评估公司的 ESG 评分。

- **风险管理**：通过分析各种风险因素制定有效的风险策略。

- **欺诈检测**：识别潜在的欺诈性交易模式，增强金融安全。

- **自动化 KYC 流程**：FinGPT 可以通过分析文档进行身份验证、与数据库交叉核对信息以及检测不一致来简化 KYC 流程。它还可以利用其 NLP 能力解释复杂的法律文件。

- **增强反洗钱（AML）措施**：FinGPT 可以成为 ML 操作中的宝贵工具。它可用于分析资金流动、识别可疑模式并突出需要进一步调查的交易。

- **低代码开发**：通过用户友好的界面促进软件创建，减少对传统编程的依赖。

- **金融教育**：作为 AI 导师简化复杂的金融概念，以提高金融素养。

通过连接这些独特但相互关联的组件，FinGPT 为利用金融 AI 提供整体且可访问的解决方案，促进金融行业的研究、创新和实际应用。

---

## 4 FinLLMs 的以数据为中心的方法

对于金融大语言模型（FinLLMs），成功的策略不仅基于模型架构的能力，还同样依赖于训练数据。我们以数据为中心的方法优先考虑收集、准备和处理金融数据。

金融数据来自各种来源，具有独特的特征。我们深入探讨不同金融数据源的具体细节，如金融新闻、公司文件和公告、社交媒体讨论和趋势。

### 4.1 金融新闻

金融新闻承载着有关世界经济、特定行业和个人公司的重要信息。此数据源通常具有以下特征：

- **及时性**：金融新闻报道及时且最新，通常捕捉金融世界的最新发展。

- **动态性**：金融新闻中包含的信息是动态的，随着经济条件和市场情绪的变化而迅速变化。

- **影响力**：金融新闻对金融市场有重大影响，影响交易者的决策，并可能导致剧烈的市场波动。

### 4.2 公司文件和公告

公司文件和公告是公司向监管机构提交的文件，提供对公司财务健康和战略方向的洞察。它们的特征是：

- **粒度**：这些文件提供有关公司财务状况的详细信息，包括资产、负债、收入和盈利能力。

- **可靠性**：公司文件包含经监管机构审查的可靠和验证数据。

- **周期性**：公司文件是周期性的，通常按季度或年度提交，提供公司财务状况的定期快照。

- **影响力**：公司公告通常对市场有重大影响，影响股票价格和投资者情绪。

### 4.3 社交媒体讨论

与金融相关的社交媒体讨论将反映公众对特定股票、行业或整体市场的情绪。这些讨论倾向于表现出：

- **变异性**：社交媒体讨论在语气、内容和质量上差异很大，使其成为丰富但复杂的信息来源。

- **实时情绪**：这些平台通常捕捉实时市场情绪，能够检测趋势和公众舆论的转变。

- **波动性**：社交媒体上表达的情绪可能高度波动，随着新闻事件或市场走势而迅速变化。

### 4.4 趋势

趋势通常通过 Seeking Alpha、Google Trends 以及其他面向金融的博客和论坛等网站观察到，提供对市场走势和投资策略的关键见解。它们的特征是：

- **分析师观点**：这些平台提供来自经验丰富的金融分析师和专家的市场预测和投资建议。

- **市场情绪**：这些平台上的论述可以反映对特定证券、行业或整体市场的集体情绪，提供对当前市场情绪的宝贵见解。

- **广泛覆盖**：趋势数据跨越不同的证券和市场板块，提供全面的市场覆盖。

这些数据来源中的每一个都提供了对金融世界的独特见解。通过整合这些不同的数据类型，FinGPT 可以促进对金融市场的全面理解，并实现有效的金融决策。

---

## 5 实验：金融情感分析

在本节中，我们评估 FinGPT 的情感分析能力。该实验展示了 FinGPT 以数据为中心的设计和轻量级适应方法在现实世界金融文本分类中的有效性。

### 5.1 数据集

我们利用通过 FinGPT 实时数据管道策划的大规模金融新闻情感数据集。该数据集包含：

- 超过 **620,000** 条清洗过的金融新闻标题；
- 来源包括 **CNBC、路透社、雅虎财经、MarketWatch** 等，通过 FinNLP 管道收集；
- 时间跨度从 **2016–2024**；
- 使用短期价格变动生成的**市场驱动标签**：

$$
\text{label} =
\begin{cases}
\text{Positive}, & r > \theta_p \\
\text{Negative}, & r < -\theta_n \\
\text{Neutral}, & |r| \leq \theta
\end{cases}
$$

其中 $r$ 表示新闻后股票的价格变化百分比。这种"自标记"方法将情感与真实市场反应对齐，并避免昂贵的手动标注。

### 5.2 模型和训练设置

我们采用轻量级的两阶段适应过程。

**基于 LoRA 的监督微调**

我们使用低秩适应（LoRA）微调预训练的 **Llama-3.1-8B-Instruct** 模型。在秩 $r=8$ 和缩放因子 $\alpha=16$ 的标准配置下，LoRA 引入的可训练参数总数约为 **8.3M**，远低于原始 80 亿参数模型的 0.1%。

微调配置如下：
- 可训练参数：8.3M
- LoRA 秩：$r=8$，缩放因子 $\alpha=16$
- 批量大小：64
- 学习率：$2 \times 10^{-4}$
- 训练轮数：3

LoRA 使 FinGPT 能够高效地获取领域特定的情感分类能力。

**股票价格强化学习（RLSP）**

为了使模型与真实市场行为保持一致，我们进一步应用 RLSP，其中环境是金融市场，奖励是股票价格在新闻后的反应。

$$R = f(\Delta p)$$

这使情感输出与实际的金融结果保持一致，并增强泛化能力。

### 5.3 基线

我们将 FinGPT 与标准金融 NLP 基线进行比较：
- FinBERT [Araci, 2019]
- BloombergGPT [Wu et al., 2023]
- ChatGPT（零样本）[OpenAI, 2023]
- Llama3.1-8B（零样本）[Grattafiori et al., 2024]

### 5.4 评估指标

我们使用以下指标评估性能：
- 准确率
- 每个类别的精确率、召回率和 F1 分数
- 宏平均 F1（减轻类别不平衡）
- 二元（正面/负面）子集的 AUC

### 5.5 结果

**整体性能**

FinGPT 显著优于所有基线，展示了以数据为中心的标记和 RLSP 强化对齐的优势。

**消融研究**

LoRA 承担了大部分工作，而 RLSP 进一步改善了市场对齐。

**案例研究**

我们使用以下标题说明模型的金融推理能力：

> "随着电动汽车竞争加剧，特斯拉再次在中国降价。"

- **人工标注**：负面（降价通常被解释为定价能力减弱和竞争压力加剧的迹象，这两者都意味着潜在的利润空间压缩，通常会引发投资者的负面情绪。）

- **基础 Llama3**：中性（模型捕捉到表面层面的措辞，但未能推断价格竞争背后的潜在金融影响。）

- **FinGPT（SFT）**：负面

- **FinGPT（RLSP）**：负面（与后续价格反应更强的对齐）

这个案例突出了 FinGPT 整合领域特定金融推理的能力，并产生与市場影響解釋更一致的情感預測。

**表 1：情感分类性能**

| 模型 | 准确率 | 宏平均 F1 | 正面 F1 | 负面 F1 | 中性 F1 |
|------|--------|----------|---------|---------|---------|
| ChatGPT (0-shot) | 63.4 | 61.7 | 64.0 | 59.1 | 62.0 |
| Llama3.1-8B (0-shot) | 57.9 | 54.4 | 56.1 | 53.2 | 54.0 |
| FinBERT | 71.2 | 69.9 | 73.0 | 69.1 | 67.5 |
| FinGPT (LoRA-SFT) | 78.8 | 77.3 | 79.6 | 76.8 | 75.4 |
| FinGPT (SFT+RLSP) | 82.1 | 80.9 | 83.4 | 81.5 | 77.8 |

**表 2：LoRA 和 RLSP 的消融**

| 配置 | 宏平均 F1 |
|------|----------|
| 基础 Llama3 | 54.4 |
| + LoRA SFT | 77.3 |
| + RLSP | 80.9 |

### 5.6 讨论

**关键观察：**

- **市场驱动标签（自标记数据）** 极大地提高了现实世界的适用性；
- **LoRA** 与全量微调相比，将适应成本降低了约 **1000 倍**；
- **RLSP** 整合了金融市场反馈，使 FinGPT 与传统监督模型区分开来。

这个实验证实，FinGPT 为金融情感分析提供了可扩展且有效的基础。

---

## 6 结论

总之，将大语言模型（LLMs）变革性地整合到金融领域带来了独特的复杂性和巨大的机遇。应对金融数据中高时间敏感性、动态金融格局和低信噪比等挑战需要有效的解决方案。FinGPT 通过利用现有的大语言模型并将其微调到特定金融应用做出了创新性响应。与 BloombergGPT 等模型相比，这种方法显著降低了适应成本和计算需求，为金融语言建模提供了更易访问、灵活且具有成本效益的解决方案。因此，它能够实现一致的更新以确保模型准确性和相关性，这在动态且时间敏感的金融世界中是一个关键方面。

---

## 7 未来工作

FinLLMs 的未来发展将专注于为金融大语言模型建立开放的行业级标准。这包括推进参数高效微调方法（如 LoRA 和 QLoRA），以支持不同金融机构的低成本、领域特定定制。此外，FinLLMs 将继续扩展其统一的数据策划管道，促进高质量、标准化的金融数据集，以简化训练和评估。通过整合开源工具、可重复的基准和透明的workflow，FinLLMs 旨在为可靠、可扩展和可互操作的金融 AI 系统提供基础。

**免责声明**：我们出于学术目的在 MIT 教育许可下分享代码。此处内容均非财务建议，也不是交易真实资金的推荐。请使用常识，并在交易或投资前务必咨询专业人士。

---

## 参考文献

[Araci, 2019] Dogu Araci. Finbert: Financial sentiment analysis with pre-trained language models. arXiv preprint arXiv:1908.10063, 2019.

[Bao et al., 2021] Siqi Bao, Huang He, Fan Wang, Hua Wu, Haifeng Wang, Wenquan Wu, Zhihua Wu, Zhen Guo, Hua Lu, Xinxian Huang, et al. Plato-xl: Exploring the large-scale pre-training of dialogue generation. arXiv preprint arXiv:2109.09519, 2021.

[Brown et al., 2020] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

[Chen et al., 2021] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122, 2021.

[DeLucia et al., 2022] Alexandra DeLucia, Shijie Wu, Aaron Mueller, Carlos Aguirre, Philip Resnik, and Mark Dredze. Bernice: a multilingual pre-trained encoder for Twitter. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 6191–6205, 2022.

[Dettmers et al., 2023] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. arXiv preprint arXiv:2305.14314, 2023.

[Devlin et al., 2018] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[Dredze et al., 2016] Mark Dredze, Prabhanjan Kambadur, Gary Kazantsev, Gideon Mann, and Miles Osborne. How twitter is changing the nature of financial news discovery. In Proceedings of the second International Workshop on Data Science for Macro-modeling, pages 1–5, 2016.

[Ethayarajh, 2019] Kawin Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. arXiv preprint arXiv:1909.00512, 2019.

[Ghosh et al., 2021] Sohom Ghosh, Shovon Sengupta, Sudip Naskar, and Sunny Kumar Singh. FinRead: A transfer learning based tool to assess readability of definitions of financial terms. In Proceedings of the 18th International Conference on Natural Language Processing (ICON), pages 658–659, National Institute of Technology Silchar, Silchar, India, December 2021. NLP Association of India (NLPAI).

[Grattafiori et al., 2024] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[Hu et al., 2021] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations, 2021.

[Jiang et al., 2023] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023.

[Lewis et al., 2019] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.

[Lewis et al., 2020] Patrick Lewis, Myle Ott, Jingfei Du, and Veselin Stoyanov. Pretrained language models for biomedical and clinical tasks: understanding and extending the state-of-the-art. In Proceedings of the 3rd Clinical Natural Language Processing Workshop, pages 146–157, 2020.

[Liu et al., 2024] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[OpenAI, 2023] OpenAI. Chatgpt. https://chat.openai.com/, 2023. Large language model accessed via ChatGPT interface.

[Ouyang et al., 2022] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

[Radford et al., 2018] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. OpenAI, 2018.

[Shah et al., 2022] Raj Sanjay Shah, Kunal Chawla, Dheeraj Eidnani, Agam Shah, Wendi Du, Sudheer Chava, Natraj Raman, Charese Smiley, Jiaao Chen, and Diyi Yang. When flue meets flang: Benchmarks and large pre-trained language model for financial domain. arXiv preprint arXiv:2211.00083, 2022.

[Sharma et al., 2022] Soumya Sharma, Tapas Nayak, Arusarka Bose, Ajay Kumar Meena, Koustuv Dasgupta, Niloy Ganguly, and Pawan Goyal. Finred: A dataset for relation extraction in financial domain. In Companion Proceedings of the Web Conference 2022, WWW '22, page 595–597, New York, NY, USA, 2022. Association for Computing Machinery.

[Team et al., 2023] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[Thoppilan et al., 2022] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.

[Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[Wu et al., 2023] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. BloombergGPT: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023.

[Yang et al., 2020] Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. Deep reinforcement learning for automated stock trading: An ensemble strategy. In Proceedings of the first ACM international conference on AI in finance, pages 1–8, 2020.

[Yang et al., 2024] Hongyang Yang, Boyu Zhang, Neng Wang, Cheng Guo, Xiaoli Zhang, Likun Lin, Junlin Wang, Tianyu Zhou, Mao Guan, Runjia Zhang, et al. Finrobot: An open-source ai agent platform for financial applications using large language models. arXiv preprint arXiv:2405.14767, 2024.

[Zhang et al., 2023] Boyu Zhang, Hongyang Yang, Tianyu Zhou, Ali Babar, and Xiao-Yang Liu. Enhancing financial sentiment analysis via retrieval augmented large language models. ACM International Conference on AI in Finance (ICAIF), 2023.

---

**翻译完成**
