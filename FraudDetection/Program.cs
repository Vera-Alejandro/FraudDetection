using Common;
using FraudDetection.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

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

            ConsoleHelper.PrintBinaryClassificationMetrics(trainerName, metrics);
        }

        private static (ITransformer model, string trainerName) TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            string[] featureColumnNames = trainingDataView.Schema.AsQueryable()
                .Select(column => column.Name)
                .Where(name => name != nameof(TransactionData.IsFraud))
                .Where(name => name != "IdPreservationColumn")
                .Where(name => name != "Type")
                .Where(name => name != "NameOrigin")
                .Where(name => name != "NameDest")
                .Where(name => name != "IsFlaggedFraud")
                .ToArray();

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", featureColumnNames)
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("TypeOneHotEncoded", "Type"))
                .Append(mlContext.Transforms.DropColumns(new string[] { "Type", "NameOrigin", "NameDest", "IsFlaggedFraud" }));

            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);

            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: nameof(TransactionData.IsFraud),
                                                                                numberOfLeaves: 25,
                                                                                numberOfIterations: 225,
                                                                                minimumExampleCountPerLeaf: 20,
                                                                                learningRate: .001);

            IDataView oneHotEncodedData = dataProcessPipeline.Fit(trainingDataView).Transform(trainingDataView);

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            PrintDataColumn(oneHotEncodedData, "TypeOneHotEncoded");

            ConsoleHelper.ConsoleWriteHeader("==========Training Model=========");

            var model = trainingPipeline.Fit(trainingDataView);

            ConsoleHelper.ConsoleWriteHeader("==========End of Training========");

            var fccPipeline = model.Append(mlContext.Transforms.CalculateFeatureContribution(model.LastTransformer).Fit(dataProcessPipeline.Fit(trainingDataView).Transform(trainingDataView)));

            return (fccPipeline, fccPipeline.ToString());
        }

        private static void PrintDataColumn(IDataView encodedData, string columnName)
        {
            var countSelectColumn = encodedData.GetColumn<float[]>(encodedData.Schema[columnName]);

            ConsoleHelper.ConsoleWriteHeader("==========One Hot Encoding Results=========");

            int j = 0;
            foreach (var row in countSelectColumn)
            {
                for (int i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]}\t");

                Console.WriteLine();

                j++;

                if (j > 10) { break; }
                {

                }
            }
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
