# Indicators of Loan Default
## by Lucrezia Morvilli


## Dataset

> I have decided to analyse one of the datasets in the options, **Prosper Loan Data**. It includes data on 100k+ loans, with extensive information and various details on them. 
> By looking at the Data Dictionary I decided to focus on 12 features of these loans, namely:
> - *ListingNumber*: The number that uniquely identifies the listing to the public as displayed on the website.
> - *ListingCreationDate*: The date the listing was created.
> - *CreditGrade*: The Credit rating that was assigned at the time the listing went live. Applicable for listings pre-2009 period and will only be populated for those listings.
> - *Term*: The length of the loan expressed in months.
> - *LoanStatus*: The current status of the loan: Cancelled,  Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue. The PastDue status will be accompanied by a delinquency bucket.
> - *ClosedDate*: Closed date is applicable for Cancelled, Completed, Chargedoff and Defaulted loan statuses.
> - *ProsperRating (numeric)*: The  Prosper Rating assigned at the time the listing was created (applicable for loans originated after July 2009): 
    -  0 - N/A
    -  1 - HR
    -  2 - E 
    -  3 - D
    -  4 - C 
    -  5 - B 
    -  6 - A 
    -  7 - AA  
> - *ProsperScore*: A custom risk score built using historical Prosper data. The score ranges from 1-10, with 10 being the best, or lowest risk score.  Applicable for loans originated after July 2009.
> - *ListingCategory*: The category of the listing that the borrower selected when posting their listing: 
    -  0 - Not Available 
    -  1 - Debt Consolidation 
    -  2 - Home Improvement 
    -  3 - Business 
    -  4 - Personal Loan 
    -  5 - Student Use 
    -  6 - Auto 
    -  7 - Other 
    -  8 - Baby&Adoption 
    -  9 - Boat 
    -  10 - Cosmetic Procedure 
    -  11 - Engagement Ring 
    -  12 - Green Loans 
    -  13 - Household Expenses 
    -  14 - Large Purchases 
    -  15 - Medical/Dental 
    -  16 - Motorcycle
    -  17 - RV 
    -  18 - Taxes 
    -  19 - Vacation
    -  20 - Wedding Loans
> - *Occupation*: The Occupation selected by the Borrower at the time they created the listing.
> - *EmploymentStatus*: The employment status of the borrower at the time they posted the listing.
> - *IsBorrowerHomeowner*: A Borrower will be classified as a homowner if they have a mortgage on their credit profile or provide documentation confirming they are a homeowner.

> **Cleaning**: The dataset is made of 112481 rows and 12 columns. It's a subset of the original ProsperLoanData dataset as I want to focus on fewer variables. I have also dropped a few rows containing an invalid value for the Prosper Score and rows having N/A for both Prosper Rating and Credit Grade.

## Summary of Findings

### Findings from Univariate Exploration

> The univariate distributions are all generally as expected, with some interesting insights that we will investigate further in the next step. 
- Term of the loan:
    - The more recent years have seen an increasing trend in number of loans being **created** and **closed**.
    - Most of the loans have a 3 year **term**.
    - The **duration** of the loans follows a bimodal distribution with peaks at the 1 year and 3 years points. 
- Credit Rating/Score:
    - By looking at the Credit Grade, Prosper Rating and Prosper Score, we found out that most of the borrowers have a **credit score** lying in the middle, ie the credit score roughly follows a normal distribution. 
- Loan features:
    - Most of the loans are current or completed, with only a very small portion being with **status** past due, defaulted or charged off. 
    - Among the **reasons** for which loans have been taken out, the most common are: Debt Consolidation, Home improvement, Business. A quarter of the reasons were classified as "Other". 
- Borrower's features:
    - As expected, most of the borrowers (ca 95%) are **employed**.
    - An interesting fact is that the population is split in half between borrowers **owning a house** and not owning a house.
    - The borrower's **occupation** is not detailed enough for most of the population, we can therefore avoid analysing this variable in the next stage. 
>    
> For the presentation, I will mention the 50-50 split of homeowners/non-homeowner and show trends for Term, Duration, Credit Score, Loan Status.

### Findings from Bivariate Exploration

> Some insights from the bivariate exploration:
   - Prosper Score and Prosper Rating are highly correlated variables.
   - People with a lower Prosper score or Credit Rating tend to have a higher chance to have their loans charge off or default. This also tells us that the rating is a good and useful indicator for Prosper, which is always good news!
   -  There is a peak of homeowners where rating is AA/A or 6/7 and the amount is decreasing with lower credit ratings. 
   - There is a smaller proportion of homeowners in the sample of defaulted and charged off loans, compared to the full population.
> One other feature of interest was the duration of the loans. **Before 2009, only 3Y loans were offered**, which obviously makes the analysis by loan term more complex. Also, a lot of defaulted/charged off loans were closed in 2009. 
> 
> Among the completed loans, ie excluding current and past due, generally speaking **1Y and 3Y loans lasted the whole term**, whereas **5Y loans were completed earlier**. 
>
> However, this might due to the fact that the first 5Y loan was created in 2010 and the dataset is until 2014 so we would **need data for the following years** as well to make conclusions on whether 5Y loans get repaid earlier or not. 
>
> Another interesting insight is that, even though the most common loan categories remain the same across the different loan terms, higher the term, **higher is the proportion of loans for Debt Consolidation**.
>
> For the presentation, I will show the following relations:
- Loan Status vs Rating
- Homeowner vs Rating
- Duration vs Term
- Duration vs Loan Status
- Term vs Loan Status
- Prosper Score vs Rating
>
> I will also mention that Prosper Score and Prosper Rating are two highly correlated variables. As these two are used after 2009, I have created a new Rating variable that picks up Credit Grade where populated and, otherwise, Prosper Rating and use these for the analyses. 

### Findings from Multivariate Exploration

>  During the last phase of this exploratory analysis, it became obvious that owning a home and credit rating are good indicators of loan default (or writing off). For instance, 61% of the HR loans defaulted/ got charged off and the majority of these were non-homeowners.
>
> I expected defaulted/ charged off loans to be closed mostly after the final term date. However, 18% of the overall population of loans is closed after the term final date whereas this is true for only 4% of the defaulted/ charged off loans. This is a sign that loans that are meant to default get identified generally earlier than the final date, which is a good thing for Prosper.
>
>Two plots will be shown for the presentation:
- Term vs Duration vs Loan Status
- Credit Rating vs Homeowner vs Loan Status

## Key Insights for Presentation

> The main thread of my presentation and analysis is the focus on indicators for loan default (or writing off). I have looked at the difference between agreed term and actual duration, at credit rating and at whether borrowers own a home. These variables were analysed individually but also in relation to one another.  