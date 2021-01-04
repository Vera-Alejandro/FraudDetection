using Microsoft.ML.Data;
using System;

namespace FraudDetection.DataStructures
{
    public interface IModelEntity
    {
        void PrintToConsole();
    }

    public class TransactionData : IModelEntity
    {
        [LoadColumn(0)]
        public float Step;

        [LoadColumn(1)]
        public string Type;

        [LoadColumn(2)]
        public float Amount;

        [LoadColumn(3)]
        public string NameOrigin;

        [LoadColumn(4)]
        public float OldBalanceOrg;

        [LoadColumn(5)]
        public float NewBalanceOrig;

        [LoadColumn(6)]
        public string NameDest;

        [LoadColumn(7)]
        public float OldBalanceDest;

        [LoadColumn(8)]
        public float NewBalanceDest;

        [LoadColumn(9)]
        public bool IsFraud;

        [LoadColumn(10)]
        public float IsFlaggedFraud;

        public void PrintToConsole()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Row View for Transaction Data            ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Step:                   {Step}           ");
            Console.WriteLine($"*       Type:                   {Type}           ");
            Console.WriteLine($"*       Amount:                 ${Amount}        ");
            Console.WriteLine($"*       NameOrigin:             {NameOrigin}     ");
            Console.WriteLine($"*       OldBalanceOrg:          ${OldBalanceOrg}  ");
            Console.WriteLine($"*       NewBalanceOrig:         ${NewBalanceOrig} ");
            Console.WriteLine($"*       NameDest:               {NameDest}       ");
            Console.WriteLine($"*       OldBalanceDest:         ${OldBalanceDest} ");
            Console.WriteLine($"*       NewBalanceDest:         ${NewBalanceDest} ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"*       IsFraud:                {IsFraud}        ");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"*       IsFlaggedFraud:         {IsFlaggedFraud} ");
            Console.WriteLine($"*************************************************");
        }
    }
}