# Data Science Salary Prediction

## Description

The "Data Science Salary Prediction" project is a data-driven initiative aimed at predicting salaries for data science positions based on various job-related features. Leveraging data science and machine learning techniques, this project seeks to provide insights into salary expectations for professionals in the data science field.

## Dataset

The dataset used for training and testing the salary prediction model was sourced from [AI Jobs - Salary Dataset](https://ai-jobs.net/salaries/download/). This dataset offers a comprehensive collection of information related to data science job positions, including details such as work year, experience level, employment type, job title, employee residence, remote work ratio, company location, and company size.

## Dataset Description
The dataset from AI Jobs - Salary Dataset is structured as a single table containing comprehensive salary information. It is organized as follows:

- **work_year**: The year in which the salary was paid.
- **experience_level**: The experience level associated with the job during the year, categorized as:
    - **EN**: Entry-level / Junior
    - **MI**: Mid-level / Intermediate
    - **SE**: Senior-level / Expert
    - **EX**: Executive-level / Director
- **employment_type**: The type of employment for the role, categorized as:
    - **PT**: Part-time
    - **FT**: Full-time
    - **CT**: Contract
    - **FL**: Freelance
- **job_title**: The specific job title or role worked during the year.
- **salary**: The total gross salary amount paid.
- **salary_currency**: The currency of the salary, represented as an ISO 4217 currency code.
- **salary_in_usd**: The salary converted to USD, calculated using exchange rates (FX rate divided by the average USD rate of the respective year) with statistical data from the BIS and central banks.
- **employee_residence**: The primary country of residence of the employee during the work year, identified by an ISO 3166 country code.
- **remote_ratio**: The extent of remote work conducted, categorized as:
    - **0**: No remote work (less than 20% remote work)
    - **50**: Partially remote/hybrid work
    - **100**: Fully remote work (more than 80% remote work)
- **company_location**: The country in which the employer's main office or contracting branch is located, represented by an ISO 3166 country code.
- **company_size**: The average number of people employed by the company during the year, categorized as:
    - **S**: Less than 50 employees (small)
    - **M**: 50 to 250 employees (medium)
    - **L**: More than 250 employees (large)

This dataset provides a comprehensive overview of salary-related information for various job roles, allowing for in-depth analysis and predictive modeling related to data science salaries.

## Project Overview

The project workflow involves data preprocessing, feature engineering, and building a salary prediction model. The primary goal is to provide data science professionals and job seekers with insights into salary expectations based on job-related factors. The model's performance is evaluated using regression metrics to ensure accurate predictions.

The project demonstrates the application of data science and machine learning techniques in the context of salary prediction, which can be valuable for both job seekers and employers in the data science field.

For more details on the project, data analysis, and model development, please refer to the project's Jupyter Notebook and codebase.


## Exploratory data analysis

### Bar Plot - Work Year Count

![barplot_work_year](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/c3e74fc3-0bcd-4d51-9a60-b449c79e6fdd)
This bar plot provides an overview of the distribution of data science job postings over different work years. The x-axis represents the work years, while the y-axis displays the count of job postings for each year.

**2020**: A small sample of job postings is visible for the year 2020, indicating a relatively lower number of data science roles during that year.

**2021**: In the year 2021, there is a noticeable increase in job postings, with approximately 200 listings. This suggests a growing demand for data science professionals during that period.

**2022**: The year 2022 shows a substantial surge in job postings, with an impressive count of around 1700. This significant rise in demand indicates a thriving job market for data science roles.

**2023**: For the year 2023, the plot illustrates a substantial peak, with nearly 5000 job postings. This peak reflects a robust and expanding job market for data science professionals.

### Bar Plot - Job Title Count
![bar_plot_job_title](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/66f1eef9-5a67-42d9-8450-19b44a2b6a12)
The "Bar Plot - Job Title Count" provides a visual representation of the distribution of job titles within the dataset. In this plot, each unique job title is represented on the X-axis, while the corresponding count of employees with that job title is shown on the Y-axis.

Key Observations:

**Variety of Job Titles**: The plot reveals a diverse range of job titles, reflecting the multifaceted nature of the dataset. There is a multitude of distinct job titles.

**Low-Frequency Job Titles**: A substantial portion of the job titles displayed in the plot are characterized by low sample counts. Many job titles appear infrequently, with sample counts less than 10. These roles may be specialized or less common within the dataset.

**High-Frequency Job Titles**: Several job titles stand out as significantly more  common. These include roles such as "Machine Learning Engineer," "Data Scientist," and "Data Engineer," among others. These high-frequency job titles have a considerably larger number of samples associated with them, suggesting that they are prevalent roles within the dataset.

**Imbalance in Job Titles**: The plot highlights an imbalance in the distribution of job titles. While certain roles have a substantial representation, others are relatively rare.

### Scatter Plot - Remote Ratio to Salary
![scatterplot_remote_ratio_salary](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/96314fcb-4af8-41d5-94cb-8ded0960d9f5)
This scatter plot visualizes the relationship between an employee's remote work ratio (the extent to which they work remotely) and their corresponding salary in USD. Each point on the plot represents an individual employee, and its position is determined by their salary (X-axis) and remote work ratio (Y-axis).

Key Observations:

**Diverse Remote Work Ratios**: The scatter plot illustrates a wide range of remote work ratios among employees in the dataset. Some employees work entirely on-site (0% remote), while others work remotely to varying degrees. This diversity in remote work arrangements is evident from the spread of data points along the Y-axis.

**Salary Variation**: The plot reveals significant variability in salaries across all remote work ratios. Employees working both 100% on-site and 100% remotely receive salaries that span a broad spectrum. This variation suggests that factors beyond remote work, such as job role, experience, or company policies, influence salary levels.

**Hybrid Work and Lower Salaries**: Notably, there is a distinct cluster of data points for employees with a remote work ratio of 50%. These employees tend to receive lower salaries compared to those working entirely on-site or remotely. This observation suggests that employees with a 50% remote work arrangement may receive reduced compensation, possibly due to company policies or market dynamics.

**Outliers**: The plot may also reveal some outliers—individuals with remote work ratios or salaries significantly deviating from the main distribution. These outliers could represent unique cases or anomalies that warrant further investigation.

### Bar Plot - Experience Level to Salary
![barplot_exp_lvl_salary](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/c87dc523-ad54-4299-a2f6-d932af1fd337)
The "Bar Plot - Experience Level to Salary" provides insights into the relationship between an employee's experience level and their corresponding salary in USD. In this plot, experience levels are categorized into four groups: Entry-level / Junior, Mid-level / Intermediate, Senior-level / Expert, and Executive-level / Director. Each bar represents the average salary for employees within a specific experience level category.

Key Observations:

**Salary Progression with Experience**: The bar plot clearly demonstrates a positive correlation between experience level and salary. As expected, employees with more experience tend to earn higher salaries.

**Entry-level / Junior Salaries**: Entry-level or junior employees typically earn the lowest salaries among the four experience levels. This is a common trend, as individuals early in their careers often receive lower compensation while gaining experience and skills.

**Mid-level / Intermediate Earnings**: Mid-level or intermediate employees earn more than their junior counterparts. This salary increase reflects the additional expertise and responsibilities that come with experience.

**Senior-level / Expert Compensation**: Senior-level or expert employees receive even higher salaries, signifying their advanced skills, knowledge, and contributions to the organization. This group typically includes employees with substantial industry experience.

**Executive-level / Director Salaries**: The plot reveals that employees at the executive-level or director positions command the highest salaries among all experience levels. This aligns with the expectation that senior leadership roles come with greater responsibilities and, consequently, higher compensation.

**Salary Disparities**: The disparities in salaries across experience levels highlight the importance of experience and expertise in determining compensation. However, it's essential to note that individual factors, such as job role and company policies, may also influence salary levels.

### Bar Plot - Employee Residence Count
![bar_plot_residence](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/7d522f46-6327-4957-b6e2-8e6c9fcb7c1a)
The "Bar Plot - Employee Residence Count" provides insights into the distribution of employee residences across different countries or regions. Each bar in the plot represents a specific country or region, and the height of the bar corresponds to the count of employees residing in that location.

Key Observations:

**Diverse Employee Residences**: The bar plot reveals that employees in the dataset come from various countries around the world. While there is a wide range of locations, it's evident that many of these locations have a relatively small number of employees represented in the dataset.

**Dominance of the United States**: The United States stands out as the dominant location of employee residence, with a significantly higher count compared to other countries or regions. This dominance reflects the dataset's composition, where a substantial portion of employees is based in the U.S.

**International Representation**: Despite the dominance of the U.S., the plot also highlights the international nature of the dataset. Employees from several countries, including Great Britain, Canada, Spain, Germany, France, and India, are present in the dataset, indicating a diverse global workforce.

**Small Sample Sizes**: For many countries or regions, the bars are relatively short, signifying that only a limited number of employees reside there. These smaller counts may be due to factors such as company operations, data collection bias, or the dataset's focus.

**Potential Geographic Insights**: The distribution of employee residences can provide insights into the geographic diversity of the dataset. It may also be valuable for understanding the representation of different regions or countries within the dataset.

**Data Collection Considerations**: When interpreting this plot, it's essential to keep in mind that the dataset's composition may not accurately reflect the real-world distribution of employee residences. Data collection methods and the dataset's source can influence the geographic representation.

### Pie Plot - Company Size Distribution
![pie_plot_company_size](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/33d5b912-849f-4312-80c3-b0dfe0ba5663)
The "Pie Plot - Company Size Distribution" provides an insightful overview of the distribution of companies within the dataset based on their size. Each slice of the pie represents a specific company size category, and the size of each slice corresponds to the proportion of companies falling into that category.

Key Observations:

**Dominance of Medium-Sized Companies**: The pie plot clearly illustrates that the majority of companies in the dataset are categorized as "M" (Medium-sized), constituting nearly **90%** of the total. This indicates that a significant portion of the dataset's companies falls within the range of 50 to 250 employees.

**Limited Presence of Small Companies**: The dataset also contains a smaller percentage of "S" (Small-sized) companies, accounting for approximately **2.4%** of the total. These are companies with less than 50 employees.

**Notable Presence of Large Companies**: While the smallest slice, "L" (Large-sized) companies still make up a notable portion of the dataset, comprising about **9.3%** of the total. These are companies with more than 250 employees.

**Implications for Workforce Composition**: The distribution of company sizes can have implications for the composition of the workforce in the dataset. The dominance of medium-sized companies suggests that a significant portion of employees may work for mid-sized organizations, which can vary widely in industry and structure.

### Bar Plot - Employment Type
![bar_plot_employment_type](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/62be5aee-6cd0-4bf9-88ef-2abb3d128f9e)
The "Bar Plot - Employment Type" offers insights into the distribution of employment types for roles within the dataset. Each bar in the plot represents a distinct employment type category, and the height of each bar corresponds to the number of roles falling into that category.

Key Observations:

**Predominance of Full-Time Employment**: The bar plot clearly highlights that the most common employment type within the dataset is "FT" (Full-time). This category significantly dominates the distribution, indicating that a substantial portion of roles in the dataset are full-time positions.

**Minimal Presence of Other Employment Types**: In contrast to full-time employment, the dataset includes very limited representation from other employment types. Employment categories such as "PT" (Part-time), "CT" (Contract), and "FL" (Freelance) have a noticeably lower number of roles associated with them.

**Implications for Employment Landscape**: The prevalence of full-time employment suggests that the majority of roles in the dataset are structured as full-time positions, which typically involve a 40-hour workweek.

**Specialized Roles**: The limited presence of other employment types, such as part-time, contract, and freelance, suggests that roles with these designations are less common in the dataset. These employment types may represent specialized or less conventional job opportunities.

### Line Plot - Average Salary Over Work Years
![line_plot_avg_salary_per_year](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/3dda0cb6-5f9f-4af1-8548-d377a2e829cc)
The "Line Plot - Average Salary Over Work Years" visualizes the trend in average salary across different work years. Each point on the line represents a specific year, while the vertical axis represents the average salary in USD.

Key Observations:

**Average Salary Fluctuations**: The line plot reveals fluctuations in average salary over the years. In 2020, the average salary was approximately **$103,000**, but it dropped to around **$99,000** in 2021. This drop may be due to a smaller amount of data available for these years, potentially causing variations in the average.

**Salary Increase in 2022**: Notably, the average salary saw a substantial increase in 2022, reaching approximately **$134,000**. This significant uptick suggests that the dataset might contain a larger volume of data for 2022, contributing to a more reliable average.

**Continued Rise in 2023**: The trend continues to climb, with an average salary of around **$155,000** in 2023. This further increase indicates a potential upward trajectory in salaries, possibly reflecting broader market trends or a larger sample size for that year.

**Data Volume Impact**: It's important to note that fluctuations in average salary, particularly in 2020 and 2021, may be influenced by the volume of data available for those years. Smaller sample sizes can lead to more significant variations in the calculated average.

**Consideration for Analysis**: When interpreting the average salary trends, it's essential to consider the dataset's temporal distribution. Variability in sample sizes across years can impact the accuracy of the average and should be taken into account in any analysis or decision-making.

### Histogram - Salary Distribution
![salary_distribution](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/14393cbb-bb20-416a-a3ea-bbdf578788d6)
The "Histogram - Salary Distribution" visualizes the distribution of salaries across the dataset. It provides insights into the frequency of salary values and their overall distribution pattern.

Key Observations:

**Bell Curve Shape**: The histogram exhibits a bell curve or Gaussian distribution pattern. This characteristic shape indicates that salaries in the dataset are distributed in a manner reminiscent of a normal distribution.

**Peak Salary**: The highest frequency of salaries, represented by the peak of the bell curve, is centered around $150,000. This suggests that a substantial number of employees in the dataset receive salaries close to this value.

**Frequency Count**: The histogram's vertical axis represents the frequency count of salary values. It reveals that the frequency of salaries near **$150,000** is approximately **1200**, indicating that a significant portion of the dataset falls within this salary range.


**Distribution Spread**: As the histogram extends away from the peak in both directions, it illustrates how salary values become less frequent as they move further from the central value of **$150,000**. This diminishing frequency is a typical feature of normal distributions.

**Interpretation**: The histogram's bell curve shape and peak around **$150,000** suggest that this salary range is a common point within the dataset. Organizations or analysts may find this information valuable for understanding the typical salary distribution within their dataset.

### Distribution Plot - Salary and Job Title
![salary_distribution_job_title](https://github.com/pmvjews/ds-salary-prediction/assets/142788709/d8b1864a-e3b4-4e79-84a6-8ea7d401f572)
The "Distribution Plot - Salary and Job Title" provides insights into the salary distribution for some of the most popular job titles in the dataset. It focuses on key positions, such as Data Analyst, Business Intelligence Analyst, Data Engineer, Data Scientist, and Machine Learning Engineer.

Key Observations for Each Job Title:

**Data Analyst**:

 - **Salary Range**: Approximately $80,000 to $120,000.
 - **Distribution**: The majority of Data Analyst salaries fall within this range, with a peak around $100,000.
 - **Outliers**: There may be a few outliers with salaries outside this range, although they are relatively rare.
 
**Business Intelligence Analyst**:
 - **Salary Range**: Typically between $100,000 and $150,000.
 - **Distribution**: Salaries for Business Intelligence Analysts show a broad distribution, with a peak around $120,000 to $130,000.
 - **Outliers**: Some outliers exist with salaries exceeding $400,000, indicating significant variation in this job role.

**Data Engineer**:

 - **Salary Range**: Typically falls in the range of $110,000 to $180,000.
 - **Distribution**: Data Engineer salaries are distributed with a peak around $140,000 to $150,000.
 - **Outliers**: Outliers with salaries above $300,000 are present, suggesting that highly skilled professionals in this role can command substantial compensation.

**Data Scientist**:

 - **Salary Range**: Approximately $130,000 to almost $200,000.
 - **Distribution**: Data Scientist salaries exhibit a distribution with a peak around $160,000 to $170,000, indicating competitive compensation.
 - **Outliers**: There are outliers with salaries exceeding $300,000 and even surpassing $400,000, indicating that experienced Data Scientists can earn exceptionally high salaries.

**Machine Learning Engineer**:

 - **Salary Range**: Typically ranges between $140,000 and $210,000.
 - **Distribution**: Salaries for Machine Learning Engineers show a distribution with a peak around $160,000 to $170,000, similar to Data Scientists.
 - **Outliers**: Some outliers have salaries exceeding $300,000 and reaching above $400,000, suggesting that highly skilled Machine Learning Engineers can command top-tier compensation.

## Model training:

In this project, I have initially trained a Linear Regression model to predict housing prices based on input features. However, the model's accuracy was insufficient for making reliable predictions. Consequently, future iterations of the project will prioritize the development of more accurate and sophisticated machine learning models to enhance predictive performance.

## Project Purpose:
The primary aim of this project was to acquire hands-on experience with PySpark, a powerful framework for big data processing and analysis. Additionally, the project provided an opportunity to develop proficiency in data visualization using the Matplotlib library. The project's goals can be summarized as follows:

**Learning PySpark**: The core objective of this project was to gain a comprehensive understanding of PySpark, a versatile tool for processing large-scale datasets. Through this project, we aimed to familiarize ourselves with PySpark's essential components, such as DataFrames, transformations, and machine learning capabilities.

**Practicing Data Visualization**: In addition to PySpark, the project provided an ideal platform to refine data visualization skills using Matplotlib. I have strived to create informative and visually appealing plots and charts to effectively convey insights from the dataset.

**Exploratory Data Analysis (EDA)**: Another key purpose of this project was to perform extensive Exploratory Data Analysis (EDA) on the provided dataset. I have aimed to uncover meaningful patterns, relationships, and trends within the data, which could potentially lead to valuable business insights.

**Enhancing Data Proficiency**: The project served as an opportunity to further enhance our proficiency in working with real-world datasets. This included data cleaning, feature engineering, and preprocessing tasks to prepare the data for analysis and modeling.


## Tools Used

- Python
- Python libraries (PySpark, Numpy, Pandas, Matplotlib, Seaborn)
- Databricks

## Conclusion:
In this project, I embarked on a journey to learn PySpark and refine my data visualization skills with Matplotlib. While the initial Linear Regression model showed promise, it is lacking accuracy highlighted the need for more advanced modeling techniques in future iterations. This project serves as a foundational step toward building robust housing price prediction models and underscores the importance of continuous improvement and exploration in my data science endeavors.
