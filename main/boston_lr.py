import sys

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bostoncount <file>", file=sys.stderr)
        sys.exit(-1)

    spark = (SparkSession
            .builder
            .appName("LR_sykang")
            .getOrCreate())

    boston_file = sys.argv[1]

    # csv 파일을 spark dataframe으로 불러오기
    boston_df = (spark.read.format("csv")
        .option("header", "true") 
        .option("inferSchema", "true") 
        .load(boston_file))
    
    boston_df.show(n=5, truncate=False)
    
    # # 특징 열을 하나의 벡터로 결합
    # assembler = VectorAssembler(
    # inputCols=["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"],
    # outputCol="features")
    
    # boston_df = assembler.transform(boston_df)
    # final_data = boston_df.select("features", "medv")
    # final_data.show(n=5, truncate=False)

    # # 데이터를 학습용과 테스트용으로 분할
    # train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    # # LinearRegression 모델을 설정하고, train_data를 사용하여 모델 학습
    # lr = LinearRegression(featuresCol="features", labelCol="medv", predictionCol="predicted_medv")
    # lr_model = lr.fit(train_data)

    # # test_data셋으로 예측 수행
    # predictions = lr_model.transform(test_data)
    # predictions.show(n=10, truncate=False)

    # # 모델 평가 (RMSE, R^2)
    # # RMSE: 회귀 모델의 예측 값과 실제 값 사이의 차이 측정 (오차의 제곱 평균을 구한 뒤, 그 값을 다시 제곱근)
    # # 값이 작을수록 모델의 예측이 실제 값에 가까움
    # evaluator = RegressionEvaluator(labelCol="medv", predictionCol="predicted_medv", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) on test data: {:.3f}".format(rmse))

    # # 모델이 데이터의 변동성을 얼마나 잘 설명하나 (값 0 ~ 1)
    # # 1에 가까울수록 모델이 데이터를 잘 설명함
    # evaluator_r2 = RegressionEvaluator(labelCol="medv", predictionCol="predicted_medv", metricName="r2")
    # r2 = evaluator_r2.evaluate(predictions)
    # print("R-squared (R2) on test data: {:.3f}".format(r2))

    # # 모델의 각 feature에 대한 coefficient
    # # coefficient 값은 해당 feature이 주택 가격에 미치는 영향을 나타냄
    # # 양수인 경우, 해당 feature가 증가할수록 주택 가격이 상승
    # # 음수인 경우 반대로 주택 가격이 하락
    # coefficients = lr_model.coefficients

    # # 데이터의 feature가 모두 0일 때의 모델의 예측 값 
    # intercept = lr_model.intercept

    # print("Coefficients: ", coefficients)
    # print("Intercept: {:.3f}".format(intercept))

    # # feature의 coefficient의 절대값을 기준으로 정렬
    # feature_importance = sorted(list(zip(boston_df.columns[:-1], map(abs, coefficients))), key=lambda x: x[1], reverse=True)

    # print("Feature Importance:")
    # for feature, importance in feature_importance:
    #     print("  {}: {:.3f}".format(feature, importance))
