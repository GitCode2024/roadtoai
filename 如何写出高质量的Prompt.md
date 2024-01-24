# 如何写出高质量的 Prompt？
>原文：[《如何写出高质量的 Prompt？》](https://baoyu.io/blog/prompt-engineering/how-to-write-high-quality-prompt)
>作者：宝玉

看到有人发《全网都在找的 GPT 最权威的 160 条指令》，其实没人记得住 160 条 Prompt，也没有必要去记 160 条 Prompt！

跟 ChatGPT 交互，最重要是掌握 Prompt 的模板或者说结构，而不需要记住那么多 Prompt。

## 一、基础用法 直接输入你希望的指令，例如：

-   “请将以下内容翻译为简体中文：”
-   “请生成以下内容的摘要：”
-   “请给 10 岁的孩子解释什么是 ChatGPT”

基本上一大半的需求就直接可以满足，如果想效果更好一点，可以为 GPT 指定一个角色，这样效果会稍微好一点。例如： “你是一位专业的英文翻译，请翻译以下内容为简体中文：”

附：为什么要指定角色？
![截屏2024-01-16 16.50.12.png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/c55b3a21-843d-42d1-86a1-f93fbc8a246d/截屏2024-01-16_16.50.12.png '截屏2024-01-16 16.50.12.png')

## 二、进阶用法

提供一到多个示例，通过示例来让 GPT 按照你期望的格式输出，比如这个例子：

> 你是一个专业翻译，擅长翻译英文到中文，但是注意双引号内的英文不翻译。 例如：
> 
> "Dichroic-Filter" - Separates light into different wavelengths to create a color separation effect.
> 
> 翻译为：
> 
> "Dichroic-Filter"- 将光分离成不同的波长以创建颜色分离效果。

### 请翻译以下内容

结合示例，基本上大部分问题都可以解决。

## 三、高级

链式思考（分多步做）+ 慢思考（打印每一步的结果）

对于一些复杂的推理过程，如果直接让 GPT 给出答案，是很容易出错的！最好是让 GPT 一步步来做，并且打印出中间步骤。在 OpenAI 官方文档里面，有一篇《[GPT 最佳实践](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-instruct-the-model-to-work-out-its-own-solution-before-rushing-to-a-conclusion)》 ，就举了一个很好的例子来给学生做助教，在收到学生的问题后，不直接给出正确或者错误的结果，而是：

> 按照这些步骤来回答用户的询问。
> 
> 第 1 步--首先找出你自己的问题解决方案。不要依赖学生的解决方案，因为它可能是不正确的。在这一步中，你的所有工作都要用三重引号 (""") 括起来。
> 
> 第 2 步--将你的解决方案与学生的解决方案进行比较，评估学生的解决方案是否正确。将你在这一步的所有工作都放在三重引号 (""") 内。
> 
> 第 3 步--如果学生犯了错误，确定你可以在不泄露答案的情况下给学生什么提示。把你在这一步的所有工作都放在三重引号 (""") 内。
> 
> 第 4 步--如果学生犯了一个错误，向学生提供上一步的提示（在三重引号之外）。不要写 "第 4 步--..."，而是写 "提示："。

当然你还可以在链式思考这个基础上加上几个示例，效果更佳。

最后，下面是一个模板，绝大部分场景都可以直接套用模板而不需要记住所谓 GPT 最权威的 160 条指令，这些指令都不会超出下面的范围。

✅ 角色、技能、个性 ✅ 目标 ✅ 具体的上下文、关键词、负面词 ✅ 输入规则 ✅ 输出规则 ✅ 输入输出的例子

有关上面的模板，可以

参考 @JefferyTatsuya [这条推文](https://twitter.com/JefferyTatsuya/status/1670204872711630848).
![截屏2024-01-16 16.55.54.png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/167385b8-9699-423c-86ea-5e458e55134b/截屏2024-01-16_16.55.54.png '截屏2024-01-16 16.55.54.png')

另外推荐几个 Prompt 开源项目参考：

-   [Prompt 提示工程指南](https://promptingguide.ai/zh)
-   \[Prompt 编写模式\] ([https://github.com/prompt-engineering/prompt-patterns](https://github.com/prompt-engineering/prompt-patterns))
-   [Awesome ChatGPT Prompts](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)

![d3f2c301-c741-4e1e-a9f0-c63899345d2f (1).png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/756d3e35-4cb5-480b-afb9-6bf54424ada3/d3f2c301-c741-4e1e-a9f0-c63899345d2f__1_.png 'd3f2c301-c741-4e1e-a9f0-c63899345d2f (1).png')
![b21ad006-d81d-47c5-8bc4-c2aa1c801a20.png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/467097af-83dd-4944-a45b-04bb50193155/b21ad006-d81d-47c5-8bc4-c2aa1c801a20.png 'b21ad006-d81d-47c5-8bc4-c2aa1c801a20.png')
![abba11e3-239e-414c-839b-3484216545d3.png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/97082f5b-4d7c-49fa-a03b-c5dd765b9afb/abba11e3-239e-414c-839b-3484216545d3.png 'abba11e3-239e-414c-839b-3484216545d3.png')


