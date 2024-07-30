from __future__ import print_function

import sys

from pyspark.sql import SparkSession

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mnmcount <file>", file=sys.stderr)
        sys.exit(-1)

    spark = (SparkSession
        .builder
        .appName("MnMCount_sykang")
        .getOrCreate())
    
    mnm_file = sys.argv[1]

    # csv 파일을 spark dataframe으로 불러오기
    mnm_df = (spark.read.format("csv")
        .option("header", "true") # csv 첫 번째 행을 헤더로
        .option("inferSchema", "true") # 데이터 타입 자동 추론 (default는 string)
        .load(mnm_file))
    
    mnm_df.show(n=5, truncate=False) # 데이터의 처음 5행 출력

    # State, Color, Count 열을 선택하여 새로운 df 생성
    # groupBy를 사용하여 State, Color 별로 데이터를 그룹화하고, Count 합산 sum
    # orderBy를 사용하여 Count값을 기준으로 내림차순 정렬
    count_mnm_df = (mnm_df.select("State", "Color", "Count")
                    .groupBy("State", "Color")
                    .sum("Count")
                    .orderBy("sum(Count)", ascending=False))

    # 처음 60행 출력
    count_mnm_df.show(n=60, truncate=False)
    print("Total Rows = %d" % (count_mnm_df.count()))

    # 캘리포니아(CA) 데이터 집계
    ca_count_mnm_df = (mnm_df.select("*")
                       .where(mnm_df.State == 'CA')
                       .groupBy("State", "Color")
                       .sum("Count")
                       .orderBy("sum(Count)", ascending=False))

    
    ca_count_mnm_df.show(n=10, truncate=False)

