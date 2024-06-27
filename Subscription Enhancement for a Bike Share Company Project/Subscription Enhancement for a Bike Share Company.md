### Subscription Enhancement for a Bike Share Company: Converting Casual Riders to Annual Members through Data-Driven Insights
A comprehensive PDF report detailing the project's goals, objectives, data sources, methodology, analysis, insights, and recommendations can be found within this folder, along with this markdown file. <br>
<br>
Tools used: SQL, Tableau & Google Slides <br>
<br>
Following are the sql queries used to extract, filter, clean, organise & transform the dataset for this project within the postgreSQL database:

#### Filtering the data: <br>
1- to extract the dataset from the database <br>
1- to extract data from the exact time frame(2022-07-01 to 2023-06-30), & <br>
2- to extract the only columns that are necessary for the analysis.


```sql
CREATE TABLE Filtered_Cyclistic_Data AS (
SELECT ride_id, rideable_type, started_at, ended_at, member_casual
FROM public."Cyclistic_Trip_Data" 
WHERE started_at > '2022-06-30 23:59:59' AND ended_at < '2023-07-01 00:00:00'
ORDER BY started_at ASC 
	)
```

#### Cleaning the data: <br>
1- eliminate errors & nulls, <br>
2- trim data for leading and trailing spaces, <br>
3- remove duplicates, & <br>
4- assign the columns with their appropriate data types <br>

```sql
CREATE TABLE cleaned_cyclistic_data AS (
SELECT DISTINCT(COALESCE(TRIM(CAST(ride_id AS VARCHAR(16))))) AS ride_id, COALESCE(TRIM(CAST(rideable_type AS VARCHAR (13)))) AS rideable_type, started_at, ended_at, COALESCE(TRIM(CAST(member_casual AS VARCHAR (6)))) AS member_casual
FROM "filtered_cyclistic_data"
WHERE (ended_at > started_at)
ORDER BY started_at ASC
	)
```
#### Organising and transforming the data: <br>
1- extracting the start time, end time, start day, end day, start month and end month of the trip <br>
2- calculating the actual duration of the trip in minutes <br>
3- assign the columns with their appropriate data types <br>

```sql

CREATE TABLE organized_cyclistic_data AS (
SELECT ride_id,
member_casual AS Membership_Type,
rideable_type AS Bike_Type,
started_at AS Start_Timestamp,
cast("started_at" as time) AS start_time,
To_Char("started_at", 'Day') AS start_day,
To_Char("started_at", 'Month') AS start_month,
ended_at AS End_Timestamp,
cast("ended_at" as time) AS end_time,
To_Char("ended_at", 'Day') AS end_day,
To_Char("ended_at", 'Month') AS end_month,
ROUND(((EXTRACT(EPOCH FROM (ended_at - started_at) ))/60), 2) AS duration_in_min
FROM "cleaned_cyclistic_data"
	)
```

### Querying Analysis Tables in PostgreSQL
(To initiate visualizations in Tableau)

#### Table 1: *Count of member & casual memberships*

```sql
SELECT member_casual, count (member_casual)
FROM  organized_cyclistic_data
GROUP BY
member_casual
```

#### Table 2: *Monthly count/ Membership type*

```sql
SELECT member_casual, start_month, count (*)
FROM organized_cyclistic_data
GROUP BY
member_casual, start_month
ORDER BY
member_casual ASC
```
#### Table 3: *Start day/ Membership type*

```sql
SELECT member_casual, start_day, count(*)
FROM organized_cyclistic_data
GROUP BY member_casual, start_day
ORDER BY member_casual
```


#### Table 4: *Calculation of Average duration in minutes*

```sql
SELECT member_casual, start_day, count(*), round(Avg(duration_in_min),2)
FROM organized_cyclistic_data
GROUP BY member_casual, start_day
ORDER BY member_casual
```

#### Thank you!
