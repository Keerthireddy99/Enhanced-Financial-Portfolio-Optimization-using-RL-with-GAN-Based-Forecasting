# Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting

### Abstract— 

The stock market is a highly volatile and risky business where losses can be devastating and making money requires forethought. As such, a well-crafted portfolio is critical to the continued success of any investor. Achieving a profitable stock spread is very difficult for the average person, however, requiring diligent observation of market trends and heaps of data. We propose a machine learning framework for portfolio optimization using Reinforcement Learning. We also use a GAN to synthesize stock price data and train the RL agent on the synthesized data and compare the results with training on real data. Advantage Actor Critic (A2C) and Deep Deterministic Policy Gradients (DDPG) RL agents are trained and their performance is compared. Our findings show that overall training on real data gave better cumulative returns than on GAN synthesized data. We also find that DDPG did not show any considerable gains over A2C and sometimes gave worse cumulative returns.

#### Keywords- 

Optimization; Generative Adversarial Networks; Reinforcement Learning; Financial Portfolio; Stocks; Machine Learning; Risk factor; Value at risk; Yahoo Finance; Assets; Tensorflow;
 
### Introduction 

Financial portfolio optimization stands as a critical challenge in the realm of finance, demanding sophisticated methodologies to efficiently allocate assets, maximize returns, and mitigate risks. Conventional portfolio optimization methods often hinge on historical data, assuming static market conditions, thereby constraining their adaptability in dynamic financial landscapes. Given the inherent stochastic nature of financial markets, influenced by an array of unpredictable factors, there is a discernible call for more advanced techniques.

Reinforcement Learning (RL) has risen as a potent paradigm in artificial intelligence and machine learning, showcasing its potential to enhance decision-making processes by learning from interactions within the environment[3]. Simultaneously, Generative Adversarial Networks (GANs) have demonstrated effectiveness in generating realistic and varied data, extending their applications to financial forecasting. The fusion of RL and GAN-based forecasting emerges as an innovative and promising strategy to amplify financial portfolio optimization.

This research endeavors to overcome the limitations of traditional portfolio optimization methods by capitalizing on the adaptability and learning capabilities of RL, coupled with GANs' proficiency in data generation. The significance of this research lies in its promise to deliver more resilient and adaptive portfolio strategies adept at navigating the intricate dynamics of dynamic financial markets.

#### Research Objectives:

##### The primary goals of this study encompass the following:

##### Integrate RL for Portfolio Optimization:
Forge and implement a reinforcement learning framework for financial portfolio optimization, empowering the algorithm to dynamically adjust and evolve in response to changing market conditions [7].

##### Utilize GANs for Financial Forecasting: 
Harness Generative Adversarial Networks to produce realistic and dynamic financial market data, thereby refining forecast accuracy and elevating the decision-making process in portfolio optimization [10].

##### Evaluate and Enhance Portfolio Performance: 
Scrutinize the performance of the RL-based portfolio optimization model, augmented by GAN-generated forecasts, in comparison to traditional methods. Discern the strengths and weaknesses of the proposed approach and fine-tune the model to attain superior outcomes.

##### Novelty of the Approach:

This research introduces an innovative methodology for financial portfolio optimization, seamlessly integrating RL and GAN-based forecasting. In contrast to traditional approaches reliant on static assumptions and historical data, the proposed model exhibits dynamic adaptability to evolving market conditions through RL. The proposed model, for the first time, integrates Generative Adversarial Networks (GANs) alongside Reinforcement Learning (RL), showcasing dynamic adaptability to evolving market conditions. GANs play a transformative role by generating authentic and diverse financial data, significantly improving the precision of forecasts. This pioneering initiative represents a departure from conventional strategies, aiming to address the challenges posed by uncertainty and non-stationarity in financial markets.. This amalgamation of advanced techniques represents a pioneering initiative to tackle the challenges posed by uncertainty and non-stationarity in financial markets. The goal is to furnish investors with more resilient and adaptive portfolio strategies. Through this inventive fusion, the research aspires to make a meaningful contribution to the ongoing evolution of portfolio optimization methodologies and elevate decision-making in the dynamic landscape of financial markets.

### Literature Review

#### Financial Portfolio Optimization:

In the world of finance, experts have explored various strategies for optimizing portfolios—finding the best mix of investments to maximize profits while minimizing risks. Classic methods, like the Modern Portfolio Theory (MPT) introduced by Markowitz, focus on spreading investments to strike a balance between risk and return [1]. Over time, researchers have delved into more sophisticated approaches, including mean-variance optimization and alternative risk-return metrics.

However, a recurring theme in these studies is their heavy reliance on past data and the assumption that market conditions are stable. This approach might not yield the best strategies in real-time situations where financial markets are dynamic[12]. Recognizing the need to adapt to changing market conditions has become a crucial aspect that requires more exploration. 

#### B. Reinforcement Learning in Finance:

Recently, there's been growing interest in using Reinforcement Learning (RL) for making financial decisions. RL has proven effective in learning optimal strategies by interacting with the financial environment [2]. This makes it particularly well-suited for the unpredictable and ever-changing nature of financial markets. Some studies have successfully applied RL to tasks like portfolio optimization, asset allocation, and risk management, showcasing its potential to outperform traditional methods in specific scenarios.

However, a gap exists in the literature regarding how RL can be combined with other advanced techniques to enhance portfolio optimization further[4]. There's a need to explore how RL and data generation methods can work together to improve decision-making in finance.

#### GAN-Based Forecasting in Finance:

Generative Adversarial Networks (GANs) have shown remarkable success in creating realistic financial data, and improving the accuracy of financial forecasts[9]. Previous research has used GANs to generate synthetic financial data, making strides in risk assessment models and addressing issues related to data scarcity.

Yet, there's a gap in the literature when it comes to integrating GANs with RL for financial portfolio optimization [6]. The potential benefits of combining GAN-based forecasting with RL methodologies to create portfolios that can adapt to changing market conditions represent an area that hasn't been explored extensively.

#### Addressing Research Gaps and Proposed Research Methodology:

The gaps identified in existing research underscore the motivation for this study. By bringing together RL and GAN-based forecasting, the goal is to tackle the limitations of traditional portfolio optimization methods that often overlook the dynamic nature of financial markets[4]. This research aims to provide new insights into making portfolios adaptable by leveraging the learning capabilities of RL and the improved forecasting accuracy enabled by GANs.

In terms of methodology, the study plans to build on existing RL frameworks designed for financial decision-making. It will incorporate data generated by GANs to refine and optimize portfolio strategies. By comparing the proposed approach with traditional methods, the research aims to offer concrete evidence of its effectiveness in adapting to changing market dynamics and achieving superior performance. This integration of RL and GAN-based forecasting represents a fresh and valuable contribution to the ever-evolving field of financial portfolio optimization methodologies.

### Methodology

#### A.  Data Sources

##### Financial Data Providers: 
Yahoo Finance:Portfolio Tracking: Yahoo Finance provides portfolio tracking tools, enabling users to monitor and manage their investments in one place. A popular platform offering historical stock prices, market data, financial news, and company-specific information.
#### B. Reinforcement Learning models

The Advantage Actor-Critic (A2C) and Deep Deterministic Policy Gradients (DDPG) algorithms represent powerful approaches in reinforcement learning, each with distinctive characteristics[11]. A2C seamlessly integrates policy-based (Actor) and value-based (Critic) methods, offering stability and high sample efficiency in tasks with discrete or continuous action spaces. Leveraging the critic's advantage estimation, A2C ensures a steady learning process, supporting online adaptation and parallelization for diverse experience collection.On the other hand, DDPG excels in addressing challenges posed by continuous action spaces. Utilizing a deterministic policy, the Q network (or Actor) directly outputs actions given a state, simplifying the learning process[5]. The deterministic policy network (Critic) evaluates selected actions, guiding the actor towards decisions expected to yield higher cumulative rewards. The incorporation of target networks, both for Q and policy, enhances stability during training by gradually updating copies of the networks, thereby mitigating temporal correlation issues. DDPG's success in applications like robotic control highlights its effectiveness in navigating continuous action spaces. However, both A2C and DDPG demand careful hyperparameter tuning for optimal performance in specific problem domains.

#### C. Reinforcement Learning implementation: 
 Environment: 
 
Determining the state space is an essential first step in developing the RL framework. This includes sentiment research, technical indicators, economic considerations, and historical stock prices, among other comprehensive representations of financial data. Concurrently, the action space is computed, which defines the allowable actions of the RL agent with respect to portfolio modifications, including the purchase, sale, and holding of different assets.

RL Algorithm Selection: 

Selecting the right RL algorithm is essential. The method's capacity to handle financial information and the complexities of decision-making will determine this choice. Due to their adaptation to financial data formats, algorithms such as Q-learning, Deep Q Networks (DQN), and Policy Gradient approaches are usually taken into consideration[8].

Training the RL Agent: 

An essential first step is organizing and preparing historical financial data that will be used to feed the RL model. Normalization and structure are ensured by data preparation for best learning outcomes. In order to educate the RL agent the best portfolio allocation techniques, past financial data must be imparted. Parameter adjustments are made during subsequent optimization in order to improve convergence and performance.

Testing and Performance Assessment: 

Key financial indicators such as Sharpe ratio, annualized return, and portfolio volatility are used to assess the RL agent's performance. These measures function as standards, evaluating how well the RL-derived methods optimize portfolios.

#### D. GAN training process:

Generating synthetic financial data tailored to each stock's market behavior requires a customized procedure in order to train Generative Adversarial Networks (GANs) for individual stocks:

The financial dataset is first separated into different subsets according to each stock. These subsets only include information relevant to that specific stock, so the GAN training that follows will accurately reflect the stock's distinct features.A distinct GAN architecture is created for every stock, consisting of the generator and the discriminator as its two main parts. The discriminator gains the ability to distinguish between created and actual data for that particular stock, while the generator attempts to create artificial financial data that mimics the market patterns of the stock.

Iterative optimization is used in the GAN training process, with the discriminator honing their ability to discern between produced and genuine financial data while the generator aims to produce data that is indistinguishable from actual financial data [9]. In order to increase the authenticity of the financial data that is produced, the models repeatedly learn from the stock-specific dataset and modify their parameters across several epochs.

Fine-tuning hyperparameters like learning rates, batch sizes, and topologies is part of the optimization process for GANs, which aims to improve convergence and provide high-quality synthetic data that closely resembles the market behavior of the stock.

The produced financial data is then evaluated and verified to guarantee that it is accurate and closely resembles actual financial data for that specific stock. In order to determine the quality and dependability of the generated data, this validation entails assessing statistical measurements and visually analyzing the data.

#### E. Flow of the project

<img width="275" alt="image" src="https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/b0e182b9-8cb0-44e2-bf5a-325f12ecbbd2">

### Experimental setup

##### A. Datasets and Preprocessing 

Datasets used in Reinforcement Learning (RL) for financial portfolio optimization encompass a wide array of financial and economic indicators. Common datasets include historical stock prices, technical indicators derived from stock data, fundamental metrics, macroeconomic indicators, and sentiment analysis data.

The preprocessing of these datasets involves several essential steps:

##### Data Cleaning: 
Handling missing values, outliers, and inconsistencies within the financial datasets to ensure data integrity.

Normalization and Scaling: Scaling numerical features to a common range, such as using normalization techniques to standardize data or scaling methods to adjust values within specific ranges.

##### Feature Engineering: 
Creating new features from existing data, using high, low, open, volume and close prices data, technical indicators for momentum, volatility, volume, and trend were added.
Covariance matrix for each day was added with the lookback period of one year.

##### Autoencoder for feature reduction: 
An autoencoder was trained to reduce prices columns and technical indicators added previously down to four features which can be used to train the RL agent along with covariance matrices for each day.

##### Train-Validation-Test Split: 
Segmenting the dataset into distinct subsets for training, validation, and testing, facilitating model training, performance validation, and evaluation.These preprocessing steps aim to refine the data, making it suitable for the RL agent's training and decision-making processes.
### Results

A.    Results obtained 
GAN Predicted Prices:

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/2e857b62-1f0a-4d93-af9b-157969c6c6ee)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/8608aadb-0bd7-4246-86f1-ada8bd7c4079)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/71d102db-e56f-46c3-a5c0-9de0aa68a5ed)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/082262f6-2f60-4d80-b900-62c85336f535)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/82a0d399-97c6-4d38-9f5a-2f5ef4309c74)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/907a0960-d85b-457c-84b4-14bd8258ca5d)

![image](https://github.com/Keerthireddy99/Enhanced-Financial-Portfolio-Optimization-using-RL-with-GAN-Based-Forecasting/assets/145499897/97b79023-d750-4010-84ed-af5e96e381ff)

### Discussion

#### A.  Generative Adversarial Networks Results : 
The analysis reveals a mixed outcome in the accuracy of stock predictions using Generative Adversarial Networks (GANs). While some stocks demonstrated accurate predictions, others did not achieve the same level of precision. Despite this variation, the GANs were generally effective in capturing general trends in stock price movements.

However, the performance of GANs notably suffered during the training phase due to backtesting. Several potential causes for this suboptimal performance have been identified. Firstly, the dataset used for training the GANs was considered relatively small, with approximately 2555 data points per stock over a 10-year period. This limitation in the size of the dataset might have hindered the models' ability to comprehensively learn and generalize the intricate patterns of stock price dynamics.

Secondly, it is acknowledged that further optimization in hyperparameters is needed. Hyperparameters play a critical role in shaping the behavior and performance of machine learning models, and fine-tuning them is essential for maximizing predictive accuracy.

Given the suboptimal performance observed with GANs, the analysis proceeded to explore reinforcement learning (RL) using regular data. The decision to transition to RL indicates a strategic shift to leverage a different paradigm that may better capture the complexities of stock market dynamics. This iterative process of analyzing, identifying limitations, and adapting methodologies reflects a common approach in the dynamic field of machine learning research. The subsequent exploration of RL using regular data suggests a continued effort to refine predictive models and enhance overall performance in stock price prediction.

#### B.  Implications 

Synthetic financial data that mimics actual market situations is produced by utilizing GANs. The RL algorithms that determine how to allocate a portfolio are trained using this data. RL agents interact with the financial data provided by GANs to learn and optimize their portfolio strategies.

This strategy has the potential to transform conventional portfolio management by bringing flexible, evidence-based decision-making techniques. It makes it possible to create more adaptable investing strategies that take into account shifting market conditions[13]. It also tackles the problem of data scarcity by offering a larger dataset for RL model training.

Nevertheless, putting this strategy into practice necessitates handling the challenges posed by complicated machine learning models, guaranteeing the accuracy and comprehensibility of combined financial data, and abiding by legal and ethical requirements in the industry. If this methodology is applied successfully, it may result in creative portfolio management techniques that adjust dynamically to market fluctuations, which might enhance risk management and boost financial market performance.

#### C. Limitations and Future Research

This research investigation underscores several key limitations and delineates promising avenues for future research within the context of predictive modeling for financial markets. One notable limitation pertains to the dataset's scale, comprising approximately 2555 data points per stock over a 10-year period, potentially impeding the models' capacity to comprehensively capture the intricate dynamics of stock price movements. Additionally, the observed suboptimal performance of Generative Adversarial Networks (GANs) during training from backtesting highlights the necessity for further refinement, particularly in hyperparameter optimization. Noteworthy is the substantial discrepancy in returns between the training and testing phases, prompting concerns about the generalization capabilities of the models.

In exploring future research directions, a deliberate and systematic approach is proposed. One avenue entails investigating the synthesis of hourly data using GANs to discern whether finer temporal granularity could yield improved results, particularly in the realm of Reinforcement Learning (RL). This represents an extension of the current research paradigm, aiming to scrutinize the efficacy of GANs in generating synthetic data at a more granular level and its subsequent impact on predictive models.

Furthermore, emphasis is placed on the critical need for hyperparameter optimization, involving a meticulous exploration of diverse configurations to enhance the overall predictive capabilities of the models. The consideration of ensemble methods, where multiple models are combined, is suggested as a potential strategy to improve robustness and generalization. Moreover, the integration of external factors, such as economic indicators and news sentiment, into the predictive models is proposed to enhance their comprehensive understanding of stock price movements.

 These proposed research directions collectively represent a commitment to advancing the state of the art in predictive modeling for financial markets, acknowledging and addressing current limitations while charting a course for innovative and impactful future investigations.
 
The observed disparity in returns between the training and testing phases underscores the importance of robust generalization capabilities in predictive models. The proposed future research directions not only aim to address current shortcomings but also pave the way for innovation in the field. The deliberate exploration of hourly data synthesis with GANs demonstrates a commitment to leveraging advanced techniques for a more nuanced understanding of stock price movements over time.

Furthermore, the suggestion to explore ensemble methods and integrate external factors represents a holistic approach to predictive modeling. Combining diverse models can enhance overall robustness, while the inclusion of external data sources broadens the context for understanding stock price influences.

In essence, this research serves as a foundational step in the ongoing evolution of predictive modeling in financial markets. By acknowledging current limitations and proposing strategic future directions, the study contributes to the continuous improvement of methodologies for understanding and forecasting stock price movements.

### Conclusion
#### A.  Summary 

In summary, this research sheds light on both the challenges and potential avenues for improvement in predictive modeling for financial markets. The limitations, such as the relatively small dataset and the suboptimal performance of Generative Adversarial Networks (GANs), highlight the intricacies involved in capturing the dynamics of stock prices. Despite these hurdles, the study outlines a roadmap for future exploration, emphasizing the need for careful hyperparameter optimization, investigation of hourly data synthesis using GANs, and the consideration of ensemble methods.

#### B. Significance 

Here are the key significances:

1. Improved Risk Management:

GAN-based forecasting, combined with RL, allows for the creation of realistic synthetic financial data. This synthetic data aids in simulating various market scenarios, enabling better risk assessment and management.

2. Adaptive Decision-Making:

RL techniques integrated with GAN-generated data enable portfolio managers to make adaptive and data-driven decisions that dynamically adjust to changing market conditions.

3. Enhanced Portfolio Performance:

By leveraging GANs for generating financial data and RL algorithms for optimization, the approach seeks to create portfolios with optimized risk-return profiles, potentially leading to improved performance compared to static or traditional strategies.

4. Handling Data Scarcity and Completeness:

In situations where historical financial data is limited or incomplete, GANs offer a solution by generating synthetic data, enabling the RL agent to learn from a wider array of scenarios and patterns.

5. Future-proofing Strategies:

Financial markets are dynamic and complex. This approach offers a framework that adapts to evolving market conditions, potentially making investment strategies more resilient and future-proof.

6. Innovation in Finance:

The integration of advanced machine learning techniques like GANs and RL in portfolio optimization represents a cutting-edge approach within the financial domain, fostering innovation and opening new avenues for research and development.

7. Potential Cost Reduction:

Optimized portfolio strategies derived from GAN-generated data, coupled with RL, may lead to reduced transaction costs and increased efficiency in managing portfolios.

### References
1. Modern Portfolio Theory (MPT)  

2. Recent Advances In Reinforcement Learning  ( Link ).

3. Robust Reinforcement Learning: A Review of Foundations and Recent Advances ( A review ).

4. Generative Adversarial Networks: ( A Literature Review ) 

5. Dynamic stock-decision ensemble strategy based on deep reinforcement learning (Link )

6. Forecasting and Classifying Financial Time Series via Generative Adversarial Networks (  LINK ). 

7. Portfolio Optimization using Reinforcement Learning (Paper Reference ).

8. Multi-Agent Reinforcement Learning ( Overview of theories and algorithms )

9. A GAN-based Approach to Vast Robust Portfolio Selection ( Finance through experiment  ).

10. Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets . 

11. Model-based Deep Reinforcement Learning for Dynamic Portfolio Optimization ( Link  ).

12. Intelligent Algorithmic Trading Strategy Using Reinforcement Learning and Directional Change ( Paper ). 

13. Formulating the Concept of an Investment Strategy Adaptable to Changes in the Market Situation ( Paper Review ).
