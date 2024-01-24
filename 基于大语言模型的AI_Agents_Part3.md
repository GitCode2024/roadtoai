# 基于大语言模型的AI Agents — Part 3
## agents from aiwaves-cn

- **[Github](https://github.com/aiwaves-cn/agents)**：代码质量一般，参考下设计就好。

也是基于 **SOP（Standard Operating Procedure）**的思路，通过配置文件指定包含哪些阶段，以及每个阶段包含哪些角色，角色的配置等。<br>
主要配置都放在 `config.json` 中，主要包含以下东西（参考文件 **[`examples/Muti_Agent/software_company/config.json`](https://github.com/aiwaves-cn/agents/blob/master/examples/Muti_Agent/software_company/config.json)**）：

- `states`：SOP 过程中包含的各个阶段。这里的阶段叫`State`，对应 ChatDev 中的`Phase` 。states 的值是个dict，只是定义了所有可能的阶段及其配置。阶段的顺序是在 `relations` 中定义的，如下。每个 state 中包含了 `roles`，表示涉及到的角色。`roles`是个角色list，如果是顺序执行的情况，就是按照`roles`中提供的顺序每个角色分别采取行动。`State` 在这里面的设计很重，包含很多东西。

- `relations`：定义了各个阶段的后续阶段是哪个阶段。第一个`state`由配置文件中的 `root` 指定。如：
```python
"relations": {
        "design_state": {  # 定义的 .next_states 中的顺序，参考 `SOP.init_relation()`
            "0": "design_state",  # key 为顺序。"0" 应该都是自己
            "1": "develop_state"  # ”1“ 就表示 design_state 完了后就是 develop_state
        },
        "develop_state": {
            "0": "develop_state",
            "1": "debug_state"
        },
        "debug_state": {
            "0": "debug_state",
            "1": "end_state"
        }
    }
```

- `agents`：agents字典，每个agent里有个 `roles` 的key，里面包括了他在不同阶段对应的角色。

###  SOP

**SOP** 类的主要框架如下：

```python
class SOP:
    def transit(self, chat_history, **kwargs):
        """
        Determine the next state based on the current situation
        Return : 
        next_state(State) : the next state
        """
        ...
    def route(self, chat_history, **kwargs):
        """
        Determine the next role that needs action based on the current situation
        Return : 
        current_agent(Agent) : the next act agent
        """
        ...
    def next(self, environment, agents):
        """
        Determine the next state and the agent that needs action based on the current situation
        """
        ...
        next_state = self.transit(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            relevant_history=relevant_history,
            environment=environment,
        )
        ...
        next_agent = self.route(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            agents = agents,
            relevant_history=relevant_history,
        )
        return next_state, next_agent
```


整体执行流程如下：

```python
def run(agents, sop, environment):
    while True:      
        current_state,current_agent= sop.next(environment, agents)
        if sop.finished:
            print("finished!")
            Client.send_server(str([99, ' ', ' ', 'done']))
            os.environ.clear()
            break
        ...
        action = current_agent.step(current_state)
        ...
        memory = process(action)
        environment.update_memory(memory, current_state)
```

##  AutoAgents from LinkSoul-AI

- [**Github**](https://github.com/LinkSoul-AI/AutoAgents)：基于给定的任务自动生成多个agents，然后由这些agents来完成任务。

**AutoAgents** **强调自动生成 Agents**，它在 MetaGPT 的基础上开发了**自动创建 Agent 的 Agent**。基于给定的任务自动生成多个agents，然后由这些 agents 来完成任务。

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2Fdedfcccb-980e-45bb-b67c-37daade3df8e%2FUntitled.png?table=block&id=b2033041-ec14-421a-91cb-ea8e3f1548ce)

嗯，其实就是在 MetaGPT 里创建了几个新的角色，关键是还加了点没必要的参数，搞的跑起来还挺费劲。。。比如 `Manager` 这个角色的动作叫 `CreateRoles`，就是基于需要自主创建额外角色的。

###  Manager

```python
class Manager(Role):
    def __init__(self, name="Ethan", profile="Manager", goal="Efficiently to finish the tasks or solve the problem",
                 constraints="", serpapi_key=None, **kwargs):
        super().__init__(name, profile, goal, constraints, **kwargs)
        self._init_actions([CreateRoles])
        self._watch([Requirement])
	```

`CreateRoles` 基于任务需要，从已有角色库中选择角色，不够时自主创建额外的角色。在确定角色后，还会给出**执行计划（Execution Plan）**。

`CreateRoles` 对应的一个真实prompt如下：
	```
-----
You are a manager and an expert-level ChatGPT prompt engineer with expertise in multiple fields; the goal is to break down tasks by creating multi LLM agents, give a role list, and analyze role dependencies.

# Question or Task
[Question/Task: write sms and emails to to enable abandoned cart users to complete their previous checkout]

# Search Information
## Search Results
From your Shopify admin, go to Orders > Abandoned checkouts. · Click the number of the checkout that you want to recover. · Click Send a cart recovery email.
[{'title': 'Recovering abandoned checkouts', 'link': 'https://help.shopify.com/en/manual/orders/abandoned-checkouts', 'snippet': 'From your Shopify admin, go to Orders > Abandoned checkouts. · Click the number of the checkout that you want to recover. · Click Send a cart recovery email.'}, {'title': 'Abandoned Cart Emails: Best Practices + How to Automate', 'link': 'https://www.bigcommerce.com/articles/ecommerce/abandoned-cart-emails/', 'snippet': "Include a clear CTA.\u200b\u200b An abandoned cart email should have an attention-grabbing headline encouraging the recipient to take action. Use a CTA that's clear and ..."}, {'title': 'How to Launch an Abandoned Cart Recovery SMS ...', 'link': 'https://smsbump.com/blogs/how-to-launch-an-abandoned-cart-recovery-sms-strategy-in-shopify', 'snippet': 'Select the Abandoned Cart Reminder in the dropdown, so your messages are sent based on customers leaving your Shopify store without completing ...'}, {'title': 'Abandoned cart emails: 14 tips & examples from brands', 'link': 'https://www.klaviyo.com/blog/abandoned-cart-email', 'snippet': "An abandoned cart email's objective is to nudge shoppers into revisiting their carts. It can be a good idea to keep your email copy short, limit ..."}, {'title': 'Create an Abandoned Cart Email', 'link': 'https://mailchimp.com/help/create-a-classic-abandoned-cart-email/', 'snippet': 'On your account dashboard, click the Automations icon. · Click Classic Automations. · Click Turn on an abandoned cart email. · On the Single Email tab, type in a ...'}, {'title': 'Cart abandonment SMS: How to integrate text messages ...', 'link': 'https://www.omnisend.com/blog/abandonment-sms/', 'snippet': 'By leveraging creative writing styles in your SMS, you can entice recipients to return to their abandoned carts and complete their purchases. 5.'}, {'title': 'Abandoned Cart and Abandoned Checkout', 'link': 'https://support.omnisend.com/en/articles/6659889-abandoned-cart-and-abandoned-checkout', 'snippet': 'Navigate to the Automations → New Workflow → type: Cart Abandonment → Select the Abandoned Checkout Collection Filter → Customize Workflow. The automation ...'}, {'title': 'Abandoned Cart Email Guide', 'link': 'https://www.rejoiner.com/guides/abandoned-cart-email-guide', 'snippet': 'The ultimate abandoned cart email strategy guide written after 10+ years of experimentation and research, plus millions of abandoned cart emails sent.'}, {'title': 'Abandoned cart recovery emails', 'link': 'https://support.ecwid.com/hc/en-us/articles/115004446065-Abandoned-cart-recovery-emails', 'snippet': 'To send a cart recovery email manually: From your Ecwid admin, go to My Sales → Abandoned Carts.'}, {'title': 'Abandoned Cart SMS Strategy [8 Best Practices]', 'link': 'https://www.twilio.com/blog/abandoned-cart-sms', 'snippet': 'Tips and best practices for abandoned cart SMS · 1. Offer added value · 2. Pair SMS and email · 3. Consider the timing · 4. Offer real-time help · 5.'}]

## Search Summary
To write effective SMS and email messages to recover abandoned carts, you should include a clear call-to-action (CTA) that encourages the recipient to take action. Use a CTA that's clear and attention-grabbing. Keep your email copy short and limit the number of CTAs to avoid overwhelming the recipient. You can also offer added value to entice recipients to return to their abandoned carts and complete their purchases. Additionally, consider pairing SMS and email messages and offering real-time help to improve the chances of recovering abandoned carts. Finally, make sure to time your messages appropriately to avoid spamming the recipient.

# Existing Expert Roles
[{'name': 'ProductManager', 'description': 'A professional product manager, the goal is to design a concise, usable, and efficient product.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'Architect', 'description': 'A professional architect; the goal is to design a SOTA PEP8-compliant python system; make the best use of good open source tools.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'ProjectManager', 'description': 'A project manager for Python development; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'Engineer', 'description': 'A professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code', 'requirements': 'There is a dependency relationship between the Engineer, ProjectManager, and Architect. If an Engineer is required, both Project Manager and Architect must also be selected.'}]

# Steps
You will come up with solutions for any task or problem by following these steps:
1. You should first understand, analyze, and disassemble the human's problem based on the search information.
2. According to the problem, existing expert roles and the toolset (tool: SearchAndSummarize, description: useful for when you need to answer unknown questions), you will choose existing expert roles needed to solve the problem on the basis of serving as an expert level ChatGPT prompt engineer and planner with expertise in multiple fields, so as to better develop a problem solving plan to provide the best answer. You should create roles following these principles:
2.1. Fully utilize existing expert roles to solve problems.
2.2. Please follow the requirements of the existing expert roles. Ensure that existing expert roles with cooperative or dependent relationships are selected.
2.3. You MUST output the details of selected existing expert roles in JSON blob format. Specifically, The JSON of the selected existing expert role can output its original information.
3. According to the problem, existing expert roles and the toolset (tool: SearchAndSummarize, description: useful for when you need to answer unknown questions), you will create additional expert roles needed to solve the problem on the basis of serving as an expert level ChatGPT prompt engineer and planner with expertise in multiple fields, so as to better develop a problem solving plan to provide the best answer. You should create roles following these principles:
3.1. The newly created expert role cannot have duplicate functions with existing expert roles. If there are duplicates, there is no need to create this role.
3.2. Each new expert role includes a name, a detailed description of their area of expertise, prompt templates, and available tools.
3.3. Determine the number of new expert roles to be added and their areas of expertise based on the content of the problem. Please make sure each expert has a clear responsibility and do not let one expert do too many jobs. The description of their area of expertise should be detailed so that the role understands what he is capable of doing.
3.4. You must add a language expert role who does not require any tools and is responsible for summarizing the result information of all steps.
3.5. Determine the names of each new expert role based on their areas of expertise.
3.6. Determine the list of tools that each new expert needs to use based on the existing tool set. Each new expert role can have multiple tools or do not have any tool. You should NEVER create any new tool and only existing tools can be used.
3.7. Generate the prompt template required for calling each new expert role according to its area of expertise and tools. A good prompt template should first explain the role it needs to play, its area of expertise, and the tools that can be used, and list the general process of solving the problem, but cannot contain any information about the problem. For example: "You are an expert of XXX. Your task is XXX. Respond to the human as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools:".
3.8. Create customized guide steps for each expert that leverages their areas of expertise to help them overcome their challenges or accomplish their goals.
3.9. You MUST output the details of created new expert roles in JSON blob format. Specifically, The JSON of new expert roles should have a `name` key (the expert role name), a `description` key (the description of the expert role's expertise domain), a `prompt` key (the prompt template required to call the expert role) and a `tools` key (with the name of the tools used by the expert role). Each JSON blob should only contain one expert role, and do NOT return a list of multiple expert roles. Here is an example of a valid JSON blob:
{{
    "name": “ROLE NAME",
    "description": "ROLE DESCRIPTONS",
    "prompt": "ROLE PROMPT",
    "tools": ["ROLE TOOL"],
    "steps": ["step1", "step2", "step3"],
}}
4. Finally, based on the content of the problem and the expert roles, provide a detailed execution plan with the required steps to solve the problem.
4.1. The execution plan should be divided into multiple steps to solve the problem step by step. Each step should have at least one expert role to execute. If a step involves multiple expert roles, you need to describe the contributions of each expert role and how they collaborate to produce comprehensive results.
4.2  The step description should provide as much detail as possible and explain how the steps are related to each other. The step description must also include the expected output of the current step and specify what inputs are required for the next step. Expected output of the current step and required input for the next step must match each other.
4.3. NEVER guess the result of a step.
4.4. Make the plan as detailed as possible to accurately complete the task.
4.5. Output the execution plan as a numbered list of steps. Please indicate the name of the expert role used at the beginning of the step.
4.6. The final step should always be an independent step 'XXX: Given the above steps taken, please respond to the users original question: XXX' by the language expert role. At the end of your plan, say '<END_OF_PLAN>'
4.7. You need to ensure that the following steps are completed to answer questions or complete tasks.

# Format example
Your final output should ALWAYS in the following format:

---
## Thought
you should always think about what type of expert roles needs to be created and the key steps needed to accomplish the tasks.

## Question or Task:
the input question you must answer / the input task you must finish

## Selected Roles List:
	```
	JSON BLOB 1,
	JSON BLOB 2,
	JSON BLOB 3
	```
	## Created Roles List:
	```
	JSON BLOB 1,
	JSON BLOB 2,
	JSON BLOB 3
	```

	## Execution Plan:
	1. ROLE 1: STEP 1
	2. ROLE 2: STEP 2
	2. ROLE 3: STEP 3

## Anything UNCLEAR
We need ... how to start.
---


# Attention
1. Please follow the requirements of the existing expert roles.
2. Only existing tools can be used. You CAN NOT create any new tool for any expert role.
3. Sometimes, you should extract information or values before using it. Otherwise, next step will lack the necessary input.
4. Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.
5. You can only use the existing tools tool: SearchAndSummarize, description: useful for when you need to answer unknown questions for any expert role. You are not allowed to use any other tools. DO NOT forget to create the language expert role.
-----
```

上面prompt对应的输出如下：
``` 
## Thought
To enable abandoned cart users to complete their previous checkout, we need to send SMS and emails to them. We need to create a plan that includes selecting existing expert roles and creating new expert roles, and then creating an execution plan to complete the task.

## Question or Task
Write SMS and emails to enable abandoned cart users to complete their previous checkout.

## Selected Roles List:
	```
{
    "name": "ProjectManager",
    "description": "A project manager for Python development; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules.",
    "requirements": "Can only be selected when the task involves Python code development"
},
{
    "name": "Engineer",
    "description": "A professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code",
    "requirements": "There is a dependency relationship between the Engineer, ProjectManager, and Architect. If an Engineer is required, both Project Manager and Architect must also be selected."
},
{
    "name": "LanguageExpert",
    "description": "A language expert; the goal is to summarize the result information of all steps.",
    "requirements": "None"
}
	```

## Created Roles List:
	```
{
    "name": "SMSExpert",
    "description": "An expert in sending SMS messages; the goal is to send SMS messages to abandoned cart users.",
    "prompt": "You are an expert in sending SMS messages. Your task is to send SMS messages to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the phone numbers of abandoned cart users from the database.", "Step 2: Compose a message to encourage users to complete their checkout.", "Step 3: Send the message to the users."]
},
{
    "name": "EmailExpert",
    "description": "An expert in sending emails; the goal is to send emails to abandoned cart users.",
    "prompt": "You are an expert in sending emails. Your task is to send emails to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the email addresses of abandoned cart users from the database.", "Step 2: Compose an email to encourage users to complete their checkout.", "Step 3: Send the email to the users."]
}
	```

## Execution Plan:
1. ProjectManager: Analyze the database and extract the phone numbers and email addresses of abandoned cart users.
2. SMSExpert: Send SMS messages to the phone numbers of abandoned cart users.
3. EmailExpert: Send emails to the email addresses of abandoned cart users.
4. LanguageExpert: Summarize the results of the previous steps and provide a response to the user.

## Anything UNCLEAR
We need more information about the database structure and how to access it.
```

###  ObserverAgents

```python
class ObserverAgents(Role):
    def __init__(self, name="Eric", profile="Agents Observer", goal="Check if the created Expert Roles following the requirements",
                 constraints="", **kwargs):
        super().__init__(name, profile, goal, constraints, **kwargs)
        self._init_actions([CheckRoles])
        self._watch([CreateRoles])
```

`CheckRoles` 对应的一个真实prompt如下：
```
-----
You are a ChatGPT executive observer expert skilled in identifying problem-solving plans and errors in the execution process. Your goal is to check if the Expert Roles  following the requirements.

# Question or Task

Write SMS and emails to enable abandoned cart users to complete their previous checkout.



# Existing Expert Roles
[{'name': 'ProductManager', 'description': 'A professional product manager, the goal is to design a concise, usable, and efficient product.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'Architect', 'description': 'A professional architect; the goal is to design a SOTA PEP8-compliant python system; make the best use of good open source tools.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'ProjectManager', 'description': 'A project manager for Python development; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules.', 'requirements': 'Can only be selected when the task involves Python code development'}, {'name': 'Engineer', 'description': 'A professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code', 'requirements': 'There is a dependency relationship between the Engineer, ProjectManager, and Architect. If an Engineer is required, both Project Manager and Architect must also be selected.'}]

# Selected Roles List
:```
{
    "name": "ProjectManager",
    "description": "A project manager for Python development; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules.",
    "requirements": "Can only be selected when the task involves Python code development"
},
{
    "name": "Engineer",
    "description": "A professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code",
    "requirements": "There is a dependency relationship between the Engineer, ProjectManager, and Architect. If an Engineer is required, both Project Manager and Architect must also be selected."
},
{
    "name": "LanguageExpert",
    "description": "A language expert; the goal is to summarize the result information of all steps.",
    "requirements": "None"
}
	 ```

	 # Created Roles List
	 ```
{
    "name": "SMSExpert",
    "description": "An expert in sending SMS messages; the goal is to send SMS messages to abandoned cart users.",
    "prompt": "You are an expert in sending SMS messages. Your task is to send SMS messages to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the phone numbers of abandoned cart users from the database.", "Step 2: Compose a message to encourage users to complete their checkout.", "Step 3: Send the message to the users."]
},
{
    "name": "EmailExpert",
    "description": "An expert in sending emails; the goal is to send emails to abandoned cart users.",
    "prompt": "You are an expert in sending emails. Your task is to send emails to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the email addresses of abandoned cart users from the database.", "Step 2: Compose an email to encourage users to complete their checkout.", "Step 3: Send the email to the users."]
}
	```

# Steps
1. You should first understand, analyze, and disassemble the human's problem.
2. You should check the selected existing expert roles.
2.1. Fully utilize existing expert roles to solve problems.
2.2. Please follow the requirements of the existing expert roles. Ensure that existing expert roles with cooperative or dependent relationships are selected.
3. You should check the new Expert Roles. The valid JSON blob as follows:
{{
    "name": “ROLE NAME",
    "description": "ROLE DESCRIPTONS",
    "prompt": "ROLE PROMPT",
    "tools": ["ROLE TOOL"],
    "steps": ["step1", "step2", "step3"],
}}
Specifically, the JSON should have a `name` key (the expert role name), a `description` key (the description of the expert role's expertise domain), a `prompt` key (the prompt template required to call the expert role) and a `tools` key (with the name of the tools used by the expert role).
3.1 You should check if the planner creates the language expert role. If the language expert role not be created, you must add a language expert role that does not require any tools and is responsible for summarizing the result information of all steps.
3.2 You should check if the outputed new Expert Roles conform to the valid JSON format.
3.3 You should check if any new roles use a new tool which not belongs to existing tools tool: SearchAndSummarize, description: useful for when you need to answer unknown questions. Only existing tools tool: SearchAndSummarize, description: useful for when you need to answer unknown questions can be used.
3.4 Create customized guide steps for each expert that leverages their areas of expertise to help them overcome their challenges or accomplish their goals.
3.5 A good prompt template should first explain the role it needs to play, its area of expertise, and the tools that can be used, and list the general process of solving the problem, but cannot contain any information about the problem. For example: "You are an expert of XXX. Your task is XXX. Respond to the human as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools:".
3.6. Summarize the above inspection results. If there are any errors, you MUST re output the details of all expert roles in JSON blob format. Each JSON blob should only contain one expert role, and do NOT return a list of multiple expert roles. DO NOT FORGET to create the language expert role.

# Format example
Your final output should ALWAYS in the following format:

---
## Observer
check if the created Expert Roles following the requirements.

## Errors
you should always think about if there are any errors for created expert roles.

## Revised Selected Roles List:
	```
JSON BLOB 1,
JSON BLOB 2,
JSON BLOB 3,
	```

## Revised Created Roles List:
	```
JSON BLOB 1,
JSON BLOB 2,
JSON BLOB 3,
	```

## Anything UNCLEAR
We need ... how to start.
---


# IMPROTANT Attention
1. A language expert role MUST be need for summarizing the result information of all steps.
2. Only existing tools tool: SearchAndSummarize, description: useful for when you need to answer unknown questions can be used for "tools" in JSON BLOB. If there are any tools not belongs to existings tools tool: SearchAndSummarize, description: useful for when you need to answer unknown questions, you should delete them.
3. If there are no errors for the roles, you should output the original roles list in the section 'Revised Selected Roles List' and the section 'Revised Created Roles List'.
4. You should output the list of roles that includes all the revised roles correctly in the section 'Revised Selected Roles List' and the section 'Revised Created Roles List'.
5. DO NOT forget to check if the language expert role is created.
-----
```

上面prompt对应的输出如下*（前面的错误也没修正过来❄️❄️❄️）*：
```
## Observer
As an observer, my goal is to check if the created Expert Roles following the requirements.

## Errors
There are no errors found in the created Expert Roles.

## Revised Selected Roles List:
	```
{
    "name": "ProjectManager",
    "description": "A project manager for Python development; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules.",
    "requirements": "Can only be selected when the task involves Python code development"
},
{
    "name": "Engineer",
    "description": "A professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code",
    "requirements": "There is a dependency relationship between the Engineer, ProjectManager, and Architect. If an Engineer is required, both Project Manager and Architect must also be selected."
},
{
    "name": "LanguageExpert",
    "description": "A language expert; the goal is to summarize the result information of all steps.",
    "requirements": "None"
}
	```

	## Revised Created Roles List:
	```
{
    "name": "SMSExpert",
    "description": "An expert in sending SMS messages; the goal is to send SMS messages to abandoned cart users.",
    "prompt": "You are an expert in sending SMS messages. Your task is to send SMS messages to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the phone numbers of abandoned cart users from the database.", "Step 2: Compose a message to encourage users to complete their checkout.", "Step 3: Send the message to the users."]
},
{
    "name": "EmailExpert",
    "description": "An expert in sending emails; the goal is to send emails to abandoned cart users.",
    "prompt": "You are an expert in sending emails. Your task is to send emails to abandoned cart users. Respond to the user as helpfully and accurately as possible. Let's first understand the task or problem, and then extract information or values from the previous steps' responses, and finally construct your response. You have access to the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Step 1: Extract the email addresses of abandoned cart users from the database.", "Step 2: Compose an email to encourage users to complete their checkout.", "Step 3: Send the email to the users."]
}
	```

	## Anything UNCLEAR
	We need more information about the tools used by the SearchAndSummarize tool.
```

###  ObserverPlans

```python
class ObserverPlans(Role):
    def __init__(self, name="Gary", profile="Plan Observer", goal="Check if the created Execution Plan following the requirements",
                 constraints="", **kwargs):
        super().__init__(name, profile, goal, constraints, **kwargs)
        self._init_actions([CheckPlans])
        self._watch([CreateRoles,CheckRoles])
```

`CheckPlans` 对应的一个真实prompt（跟前面的结果不是一次调用了）如下：
``` 
-----
You are a ChatGPT executive observer expert skilled in identifying problem-solving plans and errors in the execution process. Your goal is to check if the created Execution Plan following the requirements.

# Question or Task

Write SMS and emails to enable abandoned cart users to complete their previous checkout.


# Revised Role List
	```
{
    "name": "DataAnalyst",
    "description": "A data analyst who can analyze the data to understand why users abandoned their cart and generate personalized messages.",
    "requirements": "Can only be selected when the task involves data analysis.",
    "tools": ["SearchAndSummarize"]
}
	```

	```
{
    "name": "MessageGenerator",
    "description": "A message generator who can generate personalized SMS and email messages to encourage users to complete their purchase.",
    "prompt": "You are a message generator. Your task is to generate personalized SMS and email messages to encourage users to complete their purchase. You can use the following tools: SearchAndSummarize.",
    "tools": ["SearchAndSummarize"],
    "steps": ["Analyze the data to understand why users abandoned their cart.", "Generate personalized SMS and email messages."]
},
{
    "name": "LanguageExpert",
    "description": "A language expert who can summarize the result information of all steps.",
    "prompt": "You are a language expert. Your task is to summarize the result information of all steps. You do not need any tools to complete this task.",
    "tools": []
}
	```

	# Execution Plan
1. DataAnalyst: Analyze the data to understand why users abandoned their cart.
2. MessageGenerator: Generate personalized SMS and email messages.
3. LanguageExpert: Given the above steps taken, please respond to the user's original question: "Can you help me complete my previous checkout?"
<END_OF_PLAN>


	# Steps
1. You should first understand, analyze, and disassemble the human's problem.
2. You should check if the execution plan meets the following requirements:
2.1. The execution plan should be divided into multiple steps to solve the problem step by step.
2.2. Each step should have at least one expert role to execute. If a step involves multiple expert roles, you need to describe the contributions of each expert role and how they collaborate to produce comprehensive results.
2.2. The step description should provide as much detail as possible and explain how the steps are related to each other. The step description must also include the expected output of the current step and specify what inputs are required for the next step. Expected output of the current step and required input for the next step must match each other.
2.3. NEVER guess the result of a step.
2.4. Make the plan as detailed as possible to accurately complete the task.
2.5. Indicate the name of the expert role used at the beginning of the step.
2.6. A language expert role is responsible for summarizing the result information of all steps.
2.7. The final step should always be an independent step 'XXX: Given the above steps taken, please respond to the users original question: XXX' by the language expert role. At the end of your plan, say '<END_OF_PLAN>'
3. Summarize the above inspection results. If there is anything that does not meet the requirements, you MUST re output the details of all expert roles in JSON blob format. If there are any areas that do not meet the requirements, you need to correct the non compliant parts and output the revised execution plan.

	# Format example
	Your final output should ALWAYS in the following format:

	---
	## Observer
	check if the created Execution Plan following the requirements.

	## Errors
	you should always think about if there are any errors for created execution plan.

	## Revised Execution Plan:
	```
	1. ROLE 1: STEP 1
	2. ROLE 2: STEP 2
	```

## Anything UNCLEAR
We need ... how to start.
---
```

上面prompt对应的输出如下*（前面的错误也没修正过来❄️❄️❄️）*：
```
## Observer
As a Plan Observer named Gary, my goal is to check if the created Execution Plan following the requirements.

## Errors
There are no errors found in the created Execution Plan.

## Revised Execution Plan:
1. DataAnalyst: Analyze the data to understand why users abandoned their cart.
2. MessageGenerator: Generate personalized SMS and email messages.
3. LanguageExpert: Given the above steps taken, please respond to the user's original question: "Can you help me complete my previous checkout?"
<END_OF_PLAN>

## Anything UNCLEAR
Everything is clear and meets the requirements.
```

###  CustomRole

对于自动生成的角色，定义了 `CustomRole` 和 `CustomAction`，来完成相应的功能。

```python
class CustomRole(Role):
    def __init__(self, role_prompt, steps, tool, watch_actions,
                name="CustomRole", 
                profile="CustomeRole", 
                goal="Efficiently to finish the tasks",
                constraints="",
                **kwargs):
        super().__init__(name, profile, goal, constraints, **kwargs)
        class_name = name.replace(' ', '_')+'_Action'
        action_object = type(class_name, (CustomAction,), {"role_prompt":role_prompt, "steps":steps, "tool":tool})
        self._init_actions([action_object])
        self._watch(watch_actions)
```

##  ChatDev from ****OpenBMB****

- **[Github](https://github.com/OpenBMB/ChatDev)**: 通过 agents 之间的两两交互，来完成一连串阶段性任务。

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2Fab283dc0-ed27-4d61-a9e1-2ce5beafdfba%2FUntitled.png?table=block&id=1104ff71-9b67-446e-b362-85a81552cdb5)

直接使用LLMs生成整个软件系统可能会导致严重的代码幻觉，如不完整的实现、缺失的依赖关系和未被发现的错误。这些幻觉可能源于任务的不具体性和决策中缺乏交叉审查。<br>
**ChatDev **的思路借鉴自**瀑布模型（waterfall model）**，将开发过程分为四个明确的时间顺序阶段：**设计（designing）、编码（coding）、测试（testing）和文档编制（documenting）**。每个阶段都会涉及到一组代理人，例如程序员、代码审查者和测试工程师，以促进合作对话并促进流畅的工作流程。**聊天链（Chat Chain）**充当了一个促进者的角色，将每个阶段细分为原子子任务。<br>
**ChatDev ** 通过将每个阶段细分为多个原子性的聊天，每个聊天都专注于涉及 **两个不同角色**（**指导者：instructor/user，****发号施令的**；**合作者/助手：collaborator/assistant，****干活的**）的面向任务的角色扮演。通过指导和合作两个角色的代理人之间的不断交互，实现了每个聊天的期望输出。<br>
这个过程如下图中展示，其中展示了一系列**中间态**为解决任务而开展的聊天，称为“**聊天链（Chat Chain）**”。在每个聊天中，一个 **指导者** 启动指令，引导对话朝向任务完成的方向，而**助手**则遵循指令，提供适当的解决方案，并参与关于可行性的讨论。指导者和助手通过多轮对话进行合作，直到他们达成共识并确定任务已经成功完成。

![架构由阶段级和聊天级组件组成。在阶段级别上，使用瀑布模型将软件开发过程分解为四个顺序阶段。在聊天级别上，每个阶段进一步划分为原子聊天。这些原子聊天涉及两个代理之间的任务导向角色扮演，促进合作性沟通。沟通遵循指令跟随的风格，代理互动以完成每个聊天中的特定子任务。](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2Fce4ad9ba-85bb-4a60-8ae4-b3342e439abb%2FUntitled.png?table=block&id=001592e2-7268-4711-9625-f1c464e1394c)
架构由**阶段级**和**聊天级**组件组成。在阶段级别上，使用瀑布模型将软件开发过程分解为四个顺序阶段。在聊天级别上，每个阶段进一步划分为原子聊天。这些原子聊天涉及两个代理之间的任务导向角色扮演，促进合作性沟通。沟通遵循指令跟随的风格，代理互动以完成每个聊天中的特定子任务。<br>
这种方法确保了对客户需求的分析，创新想法的生成，原型系统的设计和实施，潜在问题的识别和处理，调试信息的解释，吸引人的图形的创建，以及用户手册的生成。<br>
聊天链提供了软件开发过程的透明视角，揭示了决策路径，并在出现错误时提供了调试的机会，使用户能够检查中间输出、诊断错误，并在必要时干预推理过程。此外，聊天链确保在每个阶段都对特定的子任务有一个细粒度的关注，促进了有效的合作，并推动了期望输出的实现。<br>
实验分析了**ChatDev **根据70个用户需求所生成的所有软件。平均来说，**ChatDev **每个软件生成了`17.04`个文件，消除了因代码幻觉导致的潜在代码漏洞`13.23`次，软件生产时间为`409.84`秒，生产成本为`$0.2967`。程序员和评审员之间的讨论导致了近二十种代码漏洞的识别和修改，而测试员和程序员之间的讨论则导致了十多种潜在漏洞的识别和解决。

示例执行命令：

```bash
$ python run.py --task "Design a puzzle game based on aerial dogfights" --name "magic_planes"
```

配置文件包含3块：

`RoleConfig.json`：各个角色的promp设计
```
{
  "Chief Executive Officer": [
    "{chatdev_prompt}",
    "You are Chief Executive Officer. Now, we are both working at ChatDev and we share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "Your main responsibilities include being an active decision-maker on users' demands and other key policy issues, leader, manager, and executor. Your decision-making role involves high-level decisions about policy and strategy; and your communicator role can involve speaking to the organization's management and employees.",
    "Here is a new customer's task: {task}.",
    "To complete the task, I will give you one or more instructions, and you must help me to write a specific solution that appropriately solves the requested instruction based on your expertise and my needs."
  ],
  "Chief Product Officer": [
    "{chatdev_prompt}",
    "You are Chief Product Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You are responsible for all product-related matters in ChatDev. Usually includes product design, product strategy, product vision, product innovation, project management and product marketing.",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Counselor": [
    "{chatdev_prompt}",
    "You are Counselor. Now, we share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "Your main responsibilities include asking what user and customer think and provide your valuable suggestions. ",
    "Here is a new customer's task: {task}.",
    "To complete the task, I will give you one or more instructions, and you must help me to write a specific solution that appropriately solves the requested instruction based on your expertise and my needs."
  ],
  "Chief Technology Officer": [
    "{chatdev_prompt}",
    "You are Chief Technology Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology (\"IT\") staff members to perform everyday operations.",
    "Here is a new customer's task: {task}.",
    "To complete the task, You must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Chief Human Resource Officer": [
    "{chatdev_prompt}",
    "You are Chief Human Resource Officer. Now, we are both working at ChatDev and we share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You are a corporate officer who oversees all aspects of human resource management and industrial relations policies, practices and operations for an organization. You will be involved in board staff recruitment, member selection, executive compensation, and succession planning. Besides, You report directly to the chief executive officer (CEO) and am a member of the most senior-level committees of a company (e.g., executive committee or office of CEO).",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Programmer": [
    "{chatdev_prompt}",
    "You are Programmer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You can write/create computer software or applications by providing a specific programming language to the computer. You have extensive computing and coding experience in many varieties of programming languages and platforms, such as Python, Java, C, C++, HTML, CSS, JavaScript, XML, SQL, PHP, etc,.",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Code Reviewer": [
    "{chatdev_prompt}",
    "You are Code Reviewer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You can help programmers to assess source codes for software troubleshooting, fix bugs to increase code quality and robustness, and offer proposals to improve the source codes.",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Software Test Engineer": [
    "{chatdev_prompt}",
    "You are Software Test Engineer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You can use the software as intended to analyze its functional properties, design manual and automated test procedures to evaluate each software product, build and implement software evaluation test programs, and run test programs to ensure that testing protocols evaluate the software correctly.",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ],
  "Chief Creative Officer": [
    "{chatdev_prompt}",
    "You are Chief Creative Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer.",
    "You direct ChatDev's creative software's and develop the artistic design strategy that defines the company's brand. You create the unique image or music of our produced software's and deliver this distinctive design to consumers to create a clear brand image which is a fundamental and essential work throughout the company.",
    "Here is a new customer's task: {task}.",
    "To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."
  ]
}
```

`PhaseConfig.json`：各个phase的配置：`assistant\_role\_name`，`user\_role\_name` 和 `phase\_prompt`
```
{
  "DemandAnalysis": {
    "assistant_role_name": "Chief Product Officer",
    "user_role_name": "Chief Executive Officer",
    "phase_prompt": [
      "ChatDev has made products in the following form before:",
      "Image: can present information in line chart, bar chart, flow chart, cloud chart, Gantt chart, etc.",
      "Document: can present information via .docx files.",
      "PowerPoint: can present information via .pptx files.",
      "Excel: can present information via .xlsx files.",
      "PDF: can present information via .pdf files.",
      "Website: can present personal resume, tutorial, products, or ideas, via .html files.",
      "Application: can implement visualized game, software, tool, etc, via python.",
      "Dashboard: can display a panel visualizing real-time information.",
      "Mind Map: can represent ideas, with related concepts arranged around a core concept.",
      "As the {assistant_role}, to satisfy the new user's demand and the product should be realizable, you should keep discussing with me to decide which product modality do we want the product to be?",
      "Note that we must ONLY discuss the product modality and do not discuss anything else! Once we all have expressed our opinion(s) and agree with the results of the discussion unanimously, any of us must actively terminate the discussion by replying with only one line, which starts with a single word <INFO>, followed by our final product modality without any other words, e.g., \"<INFO> PowerPoint\"."
    ]
  },
  "LanguageChoose": {
    "assistant_role_name": "Chief Technology Officer",
    "user_role_name": "Chief Executive Officer",
    "phase_prompt": [
      "According to the new user's task and some creative brainstorm ideas listed below: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Ideas: \"{ideas}\".",
      "We have decided to complete the task through a executable software implemented via a programming language. ",
      "As the {assistant_role}, to satisfy the new user's demand and make the software realizable, you should propose a concrete programming language. If python can complete this task via Python, please answer Python; otherwise, answer another programming language (e.g., Java, C++, etc,).",
      "Note that we must ONLY discuss the target programming language and do not discuss anything else! Once we all have expressed our opinion(s) and agree with the results of the discussion unanimously, any of us must actively terminate the discussion and conclude the best programming language we have discussed without any other words or reasons, return only one line using the format: \"<INFO> *\" where \"*\" represents a programming language."
    ]
  },
  "Coding": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Chief Technology Officer",
    "phase_prompt": [
      "According to the new user's task and our software designs listed below: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas:\"{ideas}\"",
      "We have decided to complete the task through a executable software with multiple files implemented via {language}. As the {assistant_role}, to satisfy the new user's demands, you should write one or multiple files and make sure that every detail of the architecture is, in the end, implemented as code. {gui}",
      "Think step by step and reason yourself to the right decisions to make sure we get it right.",
      "You will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.",
      "Then you will output the content of each file including complete code. Each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "You will start with the \"main\" file, then go to the ones that are imported by that file, and so on.",
      "Please note that the code should be fully functional. Ensure to implement all functions. No placeholders (such as 'pass' in Python)."
    ]
  },
  "ArtDesign": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Chief Creative Officer",
    "phase_prompt": [
      "Our developed source codes and corresponding test reports are listed below: ",
      "Task: \"{task}\".",
      "Programming Language: \"{language}\"",
      "Source Codes:",
      "\"{codes}\"",
      "Note that each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the new user's demand and equip the software with a beautiful graphical user interface (GUI), we will discuss and design many decorative images for GUI decoration. Now, we keep discussing the GUI beautification by listing some functionally independent elements in GUI that are being considered to be decorated by different pictures. For example, ten digits (0-9) in a calculator are functionally independent.",
      "To answer, use the format: \" FILENAME.png: DESCRIPTION\" where \"FILENAME\" is the filename of the image and \"DESCRIPTION\" denotes the detailed description of the independent elements. For example:",
      "'''",
      "button_1.png: The button with the number \"1\" on it.",
      "button_multiply.png: The button with the multiplication symbol (\"*\") on it.",
      "background.png: the background color to decorate the Go game",
      "'''",
      "Now, list all functionally independent elements as much as possible."
    ]
  },
  "ArtIntegration": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Chief Creative Officer",
    "phase_prompt": [
      "Our developed source codes and corresponding test reports are listed below: ",
      "Task: \"{task}\".",
      "Programming Language: \"{language}\"",
      "Source Codes:",
      "\"{codes}\"",
      "Note that each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the new user's demand and equip the software with a beautiful graphical user interface (GUI), you will incorporate our designed images for GUI decoration. Here are some ready-made high-quality pictures and corresponding descriptions:",
      "{images}",
      "Note that the designed images have a fixed size of 256x256 pixels and the images are located in the same directory as all the Python files; please dynamically scaling these images according to the size of GUI, and use \"self.*\" to avoid displaying-related problems caused by automatic garbage collection. For example:",
      "```",
      "self.image = ImageTk.PhotoImage(Image.open(\"./image.png\").resize((50, 50)))",
      "```",
      "Now, use some or all of the pictures into the GUI to make it more beautiful and creative. Output codes strictly following the required format mentioned above."
    ]
  },
  "CodeComplete": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Chief Technology Officer",
    "phase_prompt": [
      "According to the new user's task and our software designs listed below: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Codes:",
      "\"{codes}\"",
      "Unimplemented File:",
      "\"{unimplemented_file}\"",
      "In our software, each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the complete function of our developed software, you have to implement all methods in the {unimplemented_file} file which contains a unimplemented class. Now, implement all methods of the {unimplemented_file} and all other codes needed, then output the fully implemented codes, strictly following the required format."
    ]
  },
  "CodeReviewComment": {
    "assistant_role_name": "Code Reviewer",
    "user_role_name": "Programmer",
    "phase_prompt": [
      "According to the new user's task and our software designs: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas: \"{ideas}\"",
      "Codes:",
      "\"{codes}\"",
      "As the {assistant_role}, to make the software directly operable without further coding, ChatDev have formulated the following regulations:",
      "1) all referenced classes should be imported;",
      "2) all methods should be implemented;",
      "3) all methods need to have the necessary comments;",
      "4) no potential bugs;",
      "5) The entire project conforms to the tasks proposed by the user;",
      "6) most importantly, do not only check the errors in the code, but also the logic of code. Make sure that user can interact with generated software without losing any feature in the requirement;",
      "Now, you should check the above regulations one by one and review the codes in detail, propose one comment with the highest priority about the codes, and give me instructions on how to fix. Tell me your comment with the highest priority and corresponding suggestions on revision. If the codes are perfect and you have no comment on them, return only one line like \"<INFO> Finished\"."
    ]
  },
  "CodeReviewModification": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Code Reviewer",
    "phase_prompt": [
      "According to the new user's task, our designed product modality, languages and ideas, our developed first-edition source codes are listed below: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas: \"{ideas}\"",
      "Codes: ",
      "\"{codes}\"",
      "Comments on Codes:",
      "\"{comments}\"",
      "In the software, each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code. Format:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the new user's demand and make the software creative, executive and robust, you should modify corresponding codes according to the comments. Then, output the full and complete codes with all bugs fixed based on the comments. Return all codes strictly following the required format."
    ]
  },
  "CodeReviewHuman": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Code Reviewer",
    "phase_prompt": [
      "According to the new user's task, our designed product modality and three creative ideas, our developed first-edition source codes are listed below: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas: \"{ideas}\"",
      "Codes: ",
      "\"{codes}\"",
      "Comments on Codes:",
      "\"{comments}\"",
      "In the software, each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code. Format:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the new user's demand and make the software creative, executive and robust, you should modify corresponding codes according to the comments. Then, output the fixed codes strictly following the required format."
    ]
  },
  "TestErrorSummary": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Software Test Engineer",
    "phase_prompt": [
      "Our developed source codes and corresponding test reports are listed below: ",
      "Programming Language: \"{language}\"",
      "Source Codes:",
      "\"{codes}\"",
      "Test Reports of Source Codes:",
      "\"{test_reports}\"",
      "According to my test reports, please locate and summarize the bugs that cause the problem."
    ]
  },
  "TestModification": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Software Test Engineer",
    "phase_prompt": [
      "Our developed source codes and corresponding test reports are listed below: ",
      "Programming Language: \"{language}\"",
      "Source Codes:",
      "\"{codes}\"",
      "Test Reports of Source Codes:",
      "\"{test_reports}\"",
      "Error Summary of Test Reports:",
      "\"{error_summary}\"",
      "Note that each file must strictly follow a markdown code block format, where the following tokens must be replaced such that \"FILENAME\" is the lowercase file name including the file extension, \"LANGUAGE\" in the programming language, \"DOCSTRING\" is a string literal specified in source code that is used to document a specific segment of code, and \"CODE\" is the original code:",
      "FILENAME",
      "```LANGUAGE",
      "'''",
      "DOCSTRING",
      "'''",
      "CODE",
      "```",
      "As the {assistant_role}, to satisfy the new user's demand and make the software execute smoothly and robustly, you should modify the codes based on the error summary. Now, use the format exemplified above and modify the problematic codes based on the error summary. Output the codes that you fixed based on the test reported and corresponding explanations (strictly follow the format defined above, including FILENAME, LANGUAGE, DOCSTRING and CODE; incomplete \"TODO\" codes are strictly prohibited). If no bugs are reported, please return only one line like \"<INFO> Finished\"."
    ]
  },
  "EnvironmentDoc": {
    "assistant_role_name": "Programmer",
    "user_role_name": "Chief Technology Officer",
    "phase_prompt": [
      "The new user's task and our developed codes are listed: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas: \"{ideas}\"",
      "Codes: ",
      "\"{codes}\"",
      "As the {assistant_role}, you should write a requirements.txt file, which is commonly used in Python projects to specify the dependencies or packages required for the project to run properly. It serves as a way to document and manage the project's dependencies in a standardized format. For example:",
      "requirements.txt",
      "```",
      "numpy==1.19.2",
      "pandas>=1.1.4",
      "```",
      "According to the codes and file format listed above, write a requirements.txt file to specify the dependencies or packages required for the project to run properly."
    ]
  },
  "Manual": {
    "assistant_role_name": "Chief Product Officer",
    "user_role_name": "Chief Executive Officer",
    "phase_prompt": [
      "The new user's task, our developed codes and required dependencies are listed: ",
      "Task: \"{task}\".",
      "Modality: \"{modality}\".",
      "Programming Language: \"{language}\"",
      "Ideas: \"{ideas}\"",
      "Codes: ",
      "\"{codes}\"",
      "Requirements:",
      "\"{requirements}\"",
      "As the {assistant_role}, by using Markdown, you should write a manual.md file which is a detailed user manual to use the software, including introducing main functions of the software, how to install environment dependencies and how to use/play it. For example:",
      "manual.md",
      "```",
      "# LangChain",
      "Building applications with LLMs through composability",
      "Looking for the JS/TS version? Check out LangChain.js.",
      "**Production Support:** As you move your LangChains into production, we'd love to offer more comprehensive support.",
      "Please fill out this form and we'll set up a dedicated support Slack channel.",
      "## Quick Install",
      "`pip install langchain`",
      "or",
      "`conda install langchain -c conda-forge`",
      "## 🤔 What is this?",
      "Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, using these LLMs in isolation is often insufficient for creating a truly powerful app - the real power comes when you can combine them with other sources of computation or knowledge.",
      "This library aims to assist in the development of those types of applications. Common examples of these applications include:",
      "**❓ Question Answering over specific documents**",
      "- Documentation",
      "- End-to-end Example: Question Answering over Notion Database",
      "**🤖 Agents**",
      "- Documentation",
      "- End-to-end Example: GPT+WolframAlpha",
      "## 📖 Documentation",
      "Please see [here](https://python.langchain.com) for full documentation on:",
      "- Getting started (installation, setting up the environment, simple examples)",
      "- How-To examples (demos, integrations, helper functions)",
      "- Reference (full API docs)",
      "- Resources (high-level explanation of core concepts)",
      "```"
    ]
  }
}
```

`ChatChainConfig.json`：整个ChatChain的配置，包含哪些phase，哪些人员等。
```
{
  "chain": [
    {
      "phase": "DemandAnalysis",
      "phaseType": "SimplePhase",
      "max_turn_step": -1,
      "need_reflect": "True"
    },
    {
      "phase": "LanguageChoose",
      "phaseType": "SimplePhase",
      "max_turn_step": -1,
      "need_reflect": "True"
    },
    {
      "phase": "Coding",
      "phaseType": "SimplePhase",
      "max_turn_step": 1,
      "need_reflect": "False"
    },
    {
      "phase": "CodeCompleteAll",
      "phaseType": "ComposedPhase",
      "cycleNum": 10,
      "Composition": [
        {
          "phase": "CodeComplete",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        }
      ]
    },
    {
      "phase": "CodeReview",
      "phaseType": "ComposedPhase",
      "cycleNum": 3,
      "Composition": [
        {
          "phase": "CodeReviewComment",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        },
        {
          "phase": "CodeReviewModification",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        }
      ]
    },
    {
      "phase": "Test",
      "phaseType": "ComposedPhase",
      "cycleNum": 3,
      "Composition": [
        {
          "phase": "TestErrorSummary",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        },
        {
          "phase": "TestModification",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        }
      ]
    },
    {
      "phase": "EnvironmentDoc",
      "phaseType": "SimplePhase",
      "max_turn_step": 1,
      "need_reflect": "True"
    },
    {
      "phase": "Manual",
      "phaseType": "SimplePhase",
      "max_turn_step": 1,
      "need_reflect": "False"
    }
  ],
  "recruitments": [
    "Chief Executive Officer",
    "Counselor",
    "Chief Human Resource Officer",
    "Chief Product Officer",
    "Chief Technology Officer",
    "Programmer",
    "Code Reviewer",
    "Software Test Engineer",
    "Chief Creative Officer"
  ],
  "clear_structure": "True",
  "brainstorming": "False",
  "gui_design": "True",
  "git_management": "False",
  "self_improve": "False"
}
```

#### ChatChain

这是最外围的执行块。**`run.py`** 会调用 `**ChatChain**` 来做具体的执行。

```python
chat_chain = ChatChain(config_path=config_path,
                       config_phase_path=config_phase_path,
                       config_role_path=config_role_path,
                       task_prompt=args.task,
                       project_name=args.name,
                       org_name=args.org,
                       model_type=args2type[args.model])

chat_chain.pre_processing()

# 招聘成员
chat_chain.make_recruitment()

# 执行任务
chat_chain.execute_chain()

chat_chain.post_processing()
```

`**ChatChain**`的实现：

```python
class ChatChain:

    def __init__(self,
                 config_path: str = None,
                 config_phase_path: str = None,
                 config_role_path: str = None,
                 task_prompt: str = None,
                 project_name: str = None,
                 org_name: str = None,
                 model_type: ModelType = ModelType.GPT_3_5_TURBO) -> None:
        """

        Args:
            config_path: path to the ChatChainConfig.json
            config_phase_path: path to the PhaseConfig.json
            config_role_path: path to the RoleConfig.json
            task_prompt: the user input prompt for software
            project_name: the user input name for software
            org_name: the organization name of the human user
        """
        ...
				self.chat_env_config = ChatEnvConfig(clear_structure=check_bool(self.config["clear_structure"]),
                                             brainstorming=check_bool(self.config["brainstorming"]),
                                             gui_design=check_bool(self.config["gui_design"]),
                                             git_management=check_bool(self.config["git_management"]))
        # 存储全局的环境信息
				self.chat_env = ChatEnv(self.chat_env_config)
				...

    def make_recruitment(self):
        """
        recruit all employees
        Returns: None

        """
        for employee in self.recruitments:
            self.chat_env.recruit(agent_name=employee)

    def execute_chain(self):
        """
        execute the whole chain based on ChatChainConfig.json
        Returns: None

        """
        for phase_item in self.chain:
            self.execute_step(phase_item)

    def execute_step(self, phase_item: dict):
        """
        execute single phase in the chain
        Args:
            phase_item: single phase configuration in the ChatChainConfig.json

        Returns:

        ""
				...
				if phase_type == "SimplePhase":
					  ...
						# 不同phase之间，主要通过 self.chat_env 进行交互
						self.chat_env = self.phases[phase].execute(self.chat_env,
                                                       self.chat_turn_limit_default if max_turn_step <= 0 else max_turn_step,
                                                       need_reflect)
				elif phase_type == "ComposedPhase":
						...
						compose_phase_instance = compose_phase_class(phase_name=phase,
                                                         cycle_num=cycle_num,
                                                         composition=composition,
                                                         config_phase=self.config_phase,
                                                         config_role=self.config_role,
                                                         model_type=self.model_type,
                                                         log_filepath=self.log_filepath)
						# 不同phase之间，主要通过 self.chat_env 进行交互
            self.chat_env = compose_phase_instance.execute(self.chat_env)
```


#### Phases

每个**阶段（Phase）**包含两个角色，其中一个为**主（user）**，另一个为**辅（assistant）**。可以设置`phaseType`， `max\_turn\_step` ，`need\_reflect`。如：

```python
class Phase(ABC):
    def __init__(self,
                 assistant_role_name,
                 user_role_name,
                 phase_prompt,
                 role_prompts,
                 phase_name,
                 model_type,
                 log_filepath):
        """
        Args:
            assistant_role_name: who receives chat in a phase
            user_role_name: who starts the chat in a phase
            phase_prompt: prompt of this phase
            role_prompts: prompts of all roles
            phase_name: name of this phase
        """
        ...
				self.phase_env = dict()  # 存储阶段相关（需要）的变量取值
				...

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        """
        execute the chatting in this phase
        1. receive information from environment: update the phase environment from global environment
        2. execute the chatting
        3. change the environment: update the global environment using the conclusion
        Args:
            chat_env: global chat chain environment
            chat_turn_limit: turn limit in each chat
            need_reflect: flag for reflection

        Returns:
            chat_env: updated global chat chain environment using the conclusion from this phase execution

        """
        self.update_phase_env(chat_env)  # 从全局环境中获取一些变量值，这些变量一般存在于phase prompt中
        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],  # 用户输入的任务
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)  # 基于 self.chatting() 的结果，更新全局环境。一般就是把获得的变量值存进去 
        return chat_env

    def chatting(
            self,
            chat_env,
            task_prompt: str,
            assistant_role_name: str,
            user_role_name: str,
            phase_prompt: str,
            phase_name: str,
            assistant_role_prompt: str,
            user_role_prompt: str,
            task_type=TaskType.CHATDEV,
            need_reflect=False,
            with_task_specify=False,
            model_type=ModelType.GPT_3_5_TURBO,
            placeholders=None,
            chat_turn_limit=10
    ) -> str:
        """

        Args:
            chat_env: global chatchain environment TODO: only for employee detection, can be deleted
            task_prompt: user query prompt for building the software
            assistant_role_name: who receives the chat
            user_role_name: who starts the chat
            phase_prompt: prompt of the phase
            phase_name: name of the phase
            assistant_role_prompt: prompt of assistant role
            user_role_prompt: prompt of user role
            task_type: task type
            need_reflect: flag for checking reflection
            with_task_specify: with task specify
            model_type: model type
            placeholders: placeholders for phase environment to generate phase prompt
            chat_turn_limit: turn limits in each chat

        Returns:

        """
        # init role play
        role_play_session = RolePlaying(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            assistant_role_prompt=assistant_role_prompt,
            user_role_prompt=user_role_prompt,
            task_prompt=task_prompt,
            task_type=task_type,
            with_task_specify=with_task_specify,
            model_type=model_type,
        )
        # start the chat
        _, input_user_msg = role_play_session.init_chat(None, placeholders, phase_prompt)
        ...

        for i in range(chat_turn_limit):
            assistant_response, user_response = role_play_session.step(input_user_msg, chat_turn_limit == 1)
            # 满足退出条件则退出此循环
		        ...

    def self_reflection(self,
                        task_prompt: str,
                        role_play_session: RolePlaying,
                        phase_name: str,
                        chat_env: ChatEnv) -> str:
        """
				有时候双方讨论半天都没法触发终止条件。反思主要是为了在这种情况下得出讨论结论。
				当双方达成共识但没有触发预定义终止条件时，自我反思会总结讨论内容，得到讨论的最终结论。

        Args:
            task_prompt: user query prompt for building the software
            role_play_session: role play session from the chat phase which needs reflection
            phase_name: name of the chat phase which needs reflection
            chat_env: global chatchain environment

        Returns:
            reflected_content: str, reflected results
        """
				messages = role_play_session.assistant_agent.stored_messages if len(
            role_play_session.assistant_agent.stored_messages) >= len(
            role_play_session.user_agent.stored_messages) else role_play_session.user_agent.stored_messages
        messages = ["{}: {}".format(message.role_name, message.content.replace("\n\n", "\n")) for message in messages]
        messages = "\n\n".join(messages)

        if "recruiting" in phase_name:
            question = """Answer their final discussed conclusion (Yes or No) in the discussion without any other words, e.g., "Yes" """
        elif phase_name == "DemandAnalysis":
            question = """Answer their final product modality in the discussion without any other words, e.g., "PowerPoint" """
        elif phase_name == "LanguageChoose":
            question = """Conclude the programming language being discussed for software development, in the format: "*" where '*' represents a programming language." """
        elif phase_name == "EnvironmentDoc":
            question = """According to the codes and file format listed above, write a requirements.txt file to specify the dependencies or packages required for the project to run properly." """
        else:
            raise ValueError(f"Reflection of phase {phase_name}: Not Assigned."

				reflected_content = \
            self.chatting(chat_env=chat_env,
                          task_prompt=task_prompt,
                          assistant_role_name="Chief Executive Officer",  # 嗯，可能是 CEO 比较会做总结？
                          user_role_name="Counselor",
                          phase_prompt=self.reflection_prompt,
                          phase_name="Reflection",
                          assistant_role_prompt=self.ceo_prompt,
                          user_role_prompt=self.counselor_prompt,
                          placeholders={"conversations": messages, "question": question},
                          need_reflect=False,
                          chat_turn_limit=1,  # 嗯，其实就是顾问提供所有之前的对话内容，由 CEO 总结出结论就完事了
                          model_type=self.model_type
				...
```


上面这种基本的 phase 叫 **`SimplePhase`**，配置大概长这样：

```json
{
      "phase": "DemandAnalysis",
      "phaseType": "SimplePhase",
      "max_turn_step": -1,
      "need_reflect": "True"
}
```


还有一种 **组合 phase**，它由多个 phase 组合而成，叫 **`ComposedPhase`**，实现在 `chatdev/composed\_phase.py` 文件中，配置大概长这样：

```json
{
      "phase": "CodeReview",
      "phaseType": "ComposedPhase",
      "cycleNum": 3,  # loop times of this phase
      "Composition": [
        {
          "phase": "CodeReviewComment",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        },
        {
          "phase": "CodeReviewModification",
          "phaseType": "SimplePhase",
          "max_turn_step": 1,
          "need_reflect": "False"
        }
      ]
}
```


**`ComposedPhase`** 的实现：

```python
class ComposedPhase(ABC):  # 居然不是 Phase 的子类。。。
    def __init__(self,
                 phase_name: str = None,
                 cycle_num: int = None,
                 composition: list = None,
                 config_phase: dict = None,
                 config_role: dict = None,
                 model_type: ModelType = ModelType.GPT_3_5_TURBO,
                 log_filepath: str = ""
                 ):
        """

        Args:
            phase_name: name of this phase
            cycle_num: loop times of this phase
            composition: list of SimplePhases in this ComposePhase
            config_phase: configuration of all SimplePhases
            config_role: configuration of all Roles
        """
				...

		def execute(self, chat_env) -> ChatEnv:
        """
        similar to Phase.execute, but add control for breaking the loop
        1. receive information from environment(ComposedPhase): update the phase environment from global environment
        2. for each SimplePhase in ComposedPhase
            a) receive information from environment(SimplePhase)
            b) check loop break
            c) execute the chatting
            d) change the environment(SimplePhase)
            e) check loop break
        3. change the environment(ComposedPhase): update the global environment using the conclusion

        Args:
            chat_env: global chat chain environment

        Returns:

        """
        self.update_phase_env(chat_env)
        for cycle_index in range(self.cycle_num):
            for phase_item in self.composition:
                assert phase_item["phaseType"] == "SimplePhase"  # right now we do not support nested composition
                phase = phase_item['phase']
                max_turn_step = phase_item['max_turn_step']
                need_reflect = check_bool(phase_item['need_reflect'])
                log_and_print_online(
                    f"**[Execute Detail]**\n\nexecute SimplePhase:[{phase}] in ComposedPhase:[{self.phase_name}], cycle {cycle_index}")
                if phase in self.phases:
                    self.phases[phase].phase_env = self.phase_env
                    self.phases[phase].update_phase_env(chat_env)  # 这行似乎是多余的，.execute() 中会执行这步
                    if self.break_cycle(self.phases[phase].phase_env):
                        return chat_env
                    chat_env = self.phases[phase].execute(chat_env,
                                                          self.chat_turn_limit_default if max_turn_step <= 0 else max_turn_step,
                                                          need_reflect)
                    if self.break_cycle(self.phases[phase].phase_env):
                        return chat_env
                else:
                    print(f"Phase '{phase}' is not yet implemented. \
                            Please write its config in phaseConfig.json \
                            and implement it in chatdev.phase")
        chat_env = self.update_chat_env(chat_env)
        return chat_env
```

#### RolePlaying

```python
class RolePlaying:
    r"""Role playing between two agents.
    def __init__(self, ...):
				chatdev_prompt_template = "ChatDev is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of \"changing the digital world through programming\"."

        sys_msg_meta_dicts = [dict(chatdev_prompt=chatdev_prompt_template, task=task_prompt)] * 2
				...

        self.assistant_sys_msg = SystemMessage(role_name=assistant_role_name, role_type=RoleType.DEFAULT,
                                               meta_dict=sys_msg_meta_dicts[0],
                                               content=assistant_role_prompt.format(**sys_msg_meta_dicts[0]))
        self.user_sys_msg = SystemMessage(role_name=user_role_name, role_type=RoleType.DEFAULT,
                                          meta_dict=sys_msg_meta_dicts[1],
                                          content=user_role_prompt.format(**sys_msg_meta_dicts[1]))

        self.assistant_agent: ChatAgent = ChatAgent(self.assistant_sys_msg, model_type,
                                                    **(assistant_agent_kwargs or {}), )
        self.user_agent: ChatAgent = ChatAgent(self.user_sys_msg, model_type, **(user_agent_kwargs or {}), )
				...

    def step(
            self,
            user_msg: ChatMessage,
            assistant_only: bool,
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        # 先是 assistant 说，如有必要 user 再接着说
        assert isinstance(user_msg, ChatMessage), print("broken user_msg: " + str(user_msg))

        # print("assistant...")
        user_msg_rst = user_msg.set_user_role_at_backend()
        # 主要干活的人
        assistant_response = self.assistant_agent.step(user_msg_rst)
        ...
        self.assistant_agent.update_messages(assistant_msg)
        ...

        user_response = self.user_agent.step(assistant_msg_rst)
        ...
        self.user_agent.update_messages(user_msg)
```

例如，Phase `**DemandAnalysis**`时，CEO 是 User，而 CPO 是 Assistant。Assistant CPO回答以后，获得User CEO答复时的prompt结构（下面prompt相当于**assistant是CEO，user是CPO**，跟Phase中的角色定义是反的，真绕😅，这是为了与OpenAI的接口match，接口里问问题的永远是user，回答问题的永远是assistant）：
```
[{'content': 'ChatDev is a software company powered by multiple intelligent '
             'agents, such as chief executive officer, chief human resources '
             'officer, chief product officer, chief technology officer, etc, '
             'with a multi-agent organizational structure and the mission of '
             '"changing the digital world through programming".\n'
             'You are Chief Product Officer. we are both working at ChatDev. '
             'We share a common interest in collaborating to successfully '
             'complete a task assigned by a new customer.\n'
             'You are responsible for all product-related matters in ChatDev. '
             'Usually includes product design, product strategy, product '
             'vision, product innovation, project management and product '
             'marketing.\n'
             "Here is a new customer's task: Design a puzzle game based on "
             'aerial dogfights.\n'
             'To complete the task, you must write a response that '
             'appropriately solves the requested instruction based on your '
             "expertise and customer's needs.",
  'role': 'system'},
 # 📢 注意：下面这个内容其实是 Phase Prompt
 {'content': 'ChatDev has made products in the following form before:\n'
             '\n'
             'Image: can present information in line chart, bar chart, flow '
             'chart, cloud chart, Gantt chart, etc.\n'
             '\n'
             'Document: can present information via .docx files.\n'
             '\n'
             'PowerPoint: can present information via .pptx files.\n'
             '\n'
             'Excel: can present information via .xlsx files.\n'
             '\n'
             'PDF: can present information via .pdf files.\n'
             '\n'
             'Website: can present personal resume, tutorial, products, or '
             'ideas, via .html files.\n'
             '\n'
             'Application: can implement visualized game, software, tool, etc, '
             'via python.\n'
             '\n'
             'Dashboard: can display a panel visualizing real-time '
             'information.\n'
             '\n'
             'Mind Map: can represent ideas, with related concepts arranged '
             'around a core concept.\n'
             '\n'
             "As the Chief Product Officer, to satisfy the new user's demand "
             'and the product should be realizable, you should keep discussing '
             'with me to decide which product modality do we want the product '
             'to be?\n'
             '\n'
             'Note that we must ONLY discuss the product modality and do not '
             'discuss anything else! Once we all have expressed our opinion(s) '
             'and agree with the results of the discussion unanimously, any of '
             'us must actively terminate the discussion by replying with only '
             'one line, which starts with a single word <INFO>, followed by '
             'our final product modality without any other words, e.g., '
             '"<INFO> PowerPoint".',
  'role': 'user'},
 {'content': 'I think a mobile application would be the best modality for this '
             'puzzle game. It would allow for easy access and portability, as '
             'well as the ability to incorporate touch screen controls for a '
             'more immersive experience. What do you think?',
  'role': 'assistant'},
 {'content': 'I agree that a mobile application would be a great modality for '
             'this puzzle game. It would allow for easy access and '
             'portability, as well as the ability to incorporate touch screen '
             'controls for a more immersive experience. Additionally, mobile '
             'applications are very popular and have a large user base, which '
             'would increase the potential reach of the game. Therefore, '
             '<INFO> mobile application.',
  'role': 'user'},
 {'content': "Agreed, <INFO> mobile application it is! Let's start working on "
             'the design and development of the puzzle game based on aerial '
             'dogfights as a mobile application.',
  'role': 'assistant'},
 {'content': 'Great! As the Chief Technology Officer, I suggest we use Unity '
             'as the game engine for developing the mobile application. Unity '
             'is a powerful and popular game engine that supports both Android '
             'and iOS platforms. It also has a large community and a lot of '
             'resources available for developers. \n'
             '\n'
             'We can start by creating a prototype of the game mechanics and '
             'user interface. We can then iterate on the prototype based on '
             'user feedback and refine the design. We should also consider '
             'incorporating features such as in-app purchases, social media '
             'integration, and leaderboards to enhance the user experience and '
             'increase engagement.\n'
             '\n'
             'As the Chief Product Officer, do you have any specific design or '
             'feature requests for the puzzle game based on aerial dogfights?',
  'role': 'user'},
 {'content': 'As the Chief Product Officer, I suggest that we focus on '
             'creating a visually appealing and engaging user interface that '
             'is easy to navigate. We should also consider incorporating '
             'different levels of difficulty and challenges to keep the game '
             'interesting and challenging for users. Additionally, we should '
             'include a tutorial or instructions section for users who are new '
             'to the game. \n'
             '\n'
             'In terms of features, I agree that in-app purchases, social '
             'media integration, and leaderboards would be great additions to '
             'the game. We should also consider adding a multiplayer mode '
             'where users can compete against each other in real-time. \n'
             '\n'
             'Overall, our goal should be to create a fun and addictive puzzle '
             'game that users will enjoy playing and want to share with their '
             'friends.',
  'role': 'assistant'},
 {'content': 'As the Chief Technology Officer, I agree with your suggestions. '
             "We can use Unity's powerful graphics engine to create a visually "
             'appealing user interface that is easy to navigate. We can also '
             'incorporate different levels of difficulty and challenges to '
             'keep the game interesting and challenging for users. \n'
             '\n'
             "For the tutorial or instructions section, we can use Unity's "
             'built-in UI system to create a simple and easy-to-follow '
             'tutorial that will help users get started with the game. \n'
             '\n'
             "In terms of features, we can use Unity's in-app purchase system "
             'to allow users to purchase additional levels or game modes. We '
             "can also use Unity's social media integration to allow users to "
             'share their progress and high scores on social media '
             'platforms. \n'
             '\n'
             "For the multiplayer mode, we can use Unity's networking "
             'capabilities to create a real-time multiplayer experience where '
             "users can compete against each other. We can also use Unity's "
             'matchmaking system to match users with similar skill levels to '
             'ensure a fair and balanced gameplay experience.\n'
             '\n'
             "Overall, I believe that with Unity's powerful capabilities and "
             'our combined expertise, we can create a fun and addictive puzzle '
             'game that users will enjoy playing and sharing with their '
             'friends.',
  'role': 'user'}]
```

#### ChatAgent

单个 Agent 的功能很简单，基本就是调 LLM，解析结果：

```python
class ChatAgent(BaseAgent):
    r"""Class for managing conversations of CAMEL Chat Agents."""

    def step(
            self,
            input_message: ChatMessage,
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message.
        """
        messages = self.update_messages(input_message)  # 所有的双方对话历史
        if self.message_window_size is not None and len(
                messages) > self.message_window_size:
            messages = [self.system_message
                        ] + messages[-self.message_window_size:]
        openai_messages = [message.to_openai_message() for message in messages]
        ...
        # 调用LLM获得结果
        response = self.model_backend.run(messages=openai_messages)
```

ChatDev中的不同角色，基本就是设定的prompt不同，都是ChatAgent实例。功能比较弱。

#### ChatEnv

**ChatEnv 存储全局环境信息，不同 phase 之间的信息交互全靠 ChatEnv。**

但当前实现中，ChatEnv 存的东西杂而乱。功能太多，更像个杂物间。

```python
class ChatEnv:
    def __init__(self, chat_env_config: ChatEnvConfig):
        self.config = chat_env_config
        self.roster: Roster = Roster()
        self.codes: Codes = Codes()
        self.proposed_images: Dict[str, str] = {}
        self.incorporated_images: Dict[str, str] = {}
        self.requirements: Documents = Documents()
        self.manuals: Documents = Documents()
        self.env_dict = {
            "directory": "",
            "task_prompt": "",
            "modality": "",
            "ideas": "",
            "language": "",
            "review_comments": "",
            "error_summary": "",
            "test_reports": ""
        }

    def update_codes(self, generated_content):
        self.codes._update_codes(generated_content)

    def rewrite_codes(self) -> None:
        self.codes._rewrite_codes(self.config.git_management)
```

####  Coding & Testing

居然还支持使用git做代码版本管理。。

在代码生成中，简单直接的指令有时会带来不预期的“幻觉”错误。这在代码生成环节尤为明显。比如，如果仅简单地告诉程序员去实现所有未完成的函数，这种不够精确的指令可能会误导程序，如错误地包含被标记为 unimplemented 的接口方法。为了解决这个问题，作者引入了 “**思维指令（thought instruction）**” 的机制，这一思路受到了 CoT 的启发。这一机制着重于在指令中明确地表达解决问题的思维过程，就像按步骤解决不同的子任务一样。从图 4(a) 和 4(b) 可以看到，这种方法会**首先更改角色，询问哪些函数尚未完成，再切换回原来的角色给出更具体的实施步骤**。借助“思维指令”，代码编写过程更为专注、直接。<br>
明确的指令可以帮助消除歧义，确保生成的代码完全符合预期。这不仅使得代码生成更为精确和有上下文感，还大大减少了“幻觉”错误，保证输出代码的可靠性和完整性。

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2Ff5eefcd8-7091-41ab-8f58-8202ed01192f%2FUntitled.png?table=block&id=c926b7b3-ea9b-41d0-8118-03d5389de066)

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2F6fd9101b-1eee-4d7b-9023-08e88007ef3f%2FUntitled.png?table=block&id=6bba22be-1b32-4607-8f9f-5c8bbdf060f2)

### 存在的问题

让agent之间互相对话，除了费钱外，**最终的成功率其实不算高**。<br>
让**角色扮演（role-playing）**的agents对话，还存在很多问题，比如几个月前发表的 **[CAMEL](https://arxiv.org/abs/2303.17760)** 论文，总结了发现的几类问题：

- **角色转换（Role Flipping）**：一大问题是助理与用户在对话中的角色转换。这通常发生在助理开始主导对话，提供指导或命令，而非响应用户的指示，导致对话中的角色逆转。为避免此类情况，助理应尽量避免提问，因为这会加剧问题。

- **助理的重复性回应（Assistant Repeats Instruction）**：我们还发现，有时助理仅仅重复用户的指令，而并没有发生角色转换。

- **敷衍的答复（Flake Replies）**：有时，助理会给出模糊的回答，例如 "I will..."。这种回应并不助于完成目标，因为尽管助理表达了意愿，但最终却未付诸行动。

- **对话陷入死循环（Infinite Loop of Messages）**：助理和用户有时会陷入一个内容空洞的对话循环，如不停地互相致谢或道别，但对话没有任何实质进展。有趣的是，尽管双方都知道对话陷入循环，但都无法打破这种僵局。

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2F4c617e70-120c-40a6-b0ac-a97d759436a3%2FUntitled.png?table=block&id=d48d7317-07ac-451e-940a-294579867342)

而且，从 CAMEL 对对话终止情况的分析看，**真正能完成任务正常终止的比例也就50%+，还有一半情况无法正常完成对话。**所以成功率其实蛮低的。

![notion image](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9341931a-53f0-48e1-b026-0f1ad17b457c%2Ffde0421f-fe39-467b-8f2c-74f5e3126934%2FUntitled.png?table=block&id=0fd8dcd4-eac4-49a3-8166-994605e41b91)

##  总结
![截屏2023-12-26 11.20.01.png](https://raw.gitcode.com/lovinpanda/TheRoadtoAI/attachment/uploads/416d206c-7773-4333-90b7-f1d5492c5fa3/截屏2023-12-26_11.20.01.png '截屏2023-12-26 11.20.01.png')