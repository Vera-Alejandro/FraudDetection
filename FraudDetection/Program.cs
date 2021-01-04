using Common;
using Microsoft.ML;
using System;
using System.IO;

namespace FraudDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            string AssetRelativePath = @"../../../assets";
            string assetPath = GetAbsolutePath(AssetRelativePath);
            string zipDataset = Path.Combine(assetPath, "input", "SyntheticCreditCardFraud.zip");
            string fullDatasetPath = Path.Combine(assetPath, "input", "SyntheticCreditCardFraud", "credit-card-data.csv");
            string trainDatasetPath = Path.Combine(assetPath, "output", "trainData.csv");
            string testDatasetPath = Path.Combine(assetPath, "output", "testData.csv");
            string modelPath = Path.Combine(assetPath, "output", "model.zip");

            UnZipDataset(zipDataset, fullDatasetPath);

            MLContext mlContext = new MLContext();

            PrepDatasets(mlContext, fullDatasetPath, trainDatasetPath, testDatasetPath);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TransactionData>(trainDatasetPath, separatorChar: ',', hasHeader: true);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<TransactionData>(testDatasetPath, separatorChar: ',', hasHeader: true);

            (ITransformer model, string trainerName) = TrainModel(mlContext, trainingDataView);

            EvaluateModel(mlContext, model, testDataView, trainerName);

            SaveModel(mlContext, model, modelPath, trainingDataView.Schema);

            Console.WriteLine("=============== Press any key ===============");
            Console.ReadLine();
        }

        private static void SaveModel(MLContext mlContext, ITransformer model, string modelPath, DataViewSchema schema)
        {
            ConsoleHelper.ConsoleWriteHeader("===============Saving Model===============");

            mlContext.Model.Save(model, schema, modelPath);

            Console.WriteLine($"Saved model to {modelPath}");
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testDataView, string trainerName)
        {
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

            var predictions = model.Transform(testDataView);

            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions,
                                                                    labelColumnName: nameof(TransactionData.IsFraud),
                                                                    scoreColumnName: "Score");
        }

        private static (ITransformer model, string trainerName) TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            throw new NotImplementedException();
        }

        private static void PrepDatasets(MLContext mlContext, string fullDatasetPath, string trainDatasetPath, string testDatasetPath)
        {
            throw new NotImplementedException();
        }

        private static void UnZipDataset(string zipDataset, string fullDatasetPath)
        {
            throw new NotImplementedException();
        }

        private static string GetAbsolutePath(string assetRelativePath)
        {
            throw new NotImplementedException();
        }
    }
}
