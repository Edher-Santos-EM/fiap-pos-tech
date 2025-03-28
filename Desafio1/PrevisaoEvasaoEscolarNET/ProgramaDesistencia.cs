using Microsoft.ML;
using Microsoft.ML.Data;
using PrevisaoEvasaoEscolar.Models;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Reflection;

namespace PrevisaoEvasaoEscolar;

internal class ProgramaDesistencia
{
    private MLContext mlContext = null!;

    public void Execute()
    {
        mlContext = new MLContext(seed: 42);
        (IDataView dadosTreino, IDataView dadosTeste) = CarregarDados("dropout-inaugural.csv");
        (ITransformer model, IDataView trainingDataView) = TreinarAvaliarModelos(dadosTreino, dadosTeste);
        mlContext.Model.Save(model, trainingDataView.Schema, "dropout-model.zip");
    }

    /// <summary>
    /// Carrega os dados do CSV e realiza análise exploratória
    /// </summary>
    private (IDataView TrainingData, IDataView TestData) CarregarDados(string caminhoArquivo)
    {
        Console.WriteLine($"Carregando dados do arquivo: {caminhoArquivo}");

        // Carrega o dataset
        IDataView dadosCompletos = mlContext.Data.LoadFromTextFile<DadosAluno>(
            path: caminhoArquivo,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true,
            allowSparse: false);

        var preprocessingPipeline = mlContext.Transforms
            .CustomMapping<DadosAluno, DadosAlunoProcessado>((input, output) =>
                {
                    MapearParaProcessado(input, ref output);
                }, "DadosAlunoProcessadoMapping");

        IDataView dadosProcessados = preprocessingPipeline.Fit(dadosCompletos).Transform(dadosCompletos);

        var alunos = mlContext.Data.CreateEnumerable<DadosAlunoProcessado>(dadosProcessados, reuseRowObject: false).ToList();
        var estatisticasIdade = new
        {
            Media = alunos.Average(a => a.InflationRate),
            Min = alunos.Min(a => a.InflationRate),
            Max = alunos.Max(a => a.InflationRate)
        };

        // Divide em treino (80%) e teste (20%)
        DataOperationsCatalog.TrainTestData divisaoDados = mlContext.Data.TrainTestSplit(dadosProcessados, testFraction: 0.2, seed: 42);

        Console.WriteLine("Dados carregados e divididos em treino e teste com sucesso!");

        return (divisaoDados.TrainSet, divisaoDados.TestSet);
    }

    private static void MapearParaProcessado(DadosAluno aluno, ref DadosAlunoProcessado dadosAlunoProcessado)
    {
        dadosAlunoProcessado.MaritalStatus = aluno.MaritalStatus;
        dadosAlunoProcessado.ApplicationMode = aluno.ApplicationMode;
        dadosAlunoProcessado.ApplicationOrder = aluno.ApplicationOrder;
        dadosAlunoProcessado.Course = aluno.Course;
        dadosAlunoProcessado.DaytimeEveningAttendance = aluno.DaytimeEveningAttendance;
        dadosAlunoProcessado.PreviousQualification = aluno.PreviousQualification;
        dadosAlunoProcessado.Nacionality = aluno.Nacionality;
        dadosAlunoProcessado.MothersQualification = aluno.MothersQualification;
        dadosAlunoProcessado.FathersQualification = aluno.FathersQualification;
        dadosAlunoProcessado.MothersOccupation = aluno.MothersOccupation;
        dadosAlunoProcessado.FathersOccupation = aluno.FathersOccupation;
        dadosAlunoProcessado.Displaced = aluno.Displaced;
        dadosAlunoProcessado.EducationalSpecialNeeds = aluno.EducationalSpecialNeeds;
        dadosAlunoProcessado.Debtor = aluno.Debtor;
        dadosAlunoProcessado.TuitionFeesUpToDate = aluno.TuitionFeesUpToDate;
        dadosAlunoProcessado.Gender = aluno.Gender;
        dadosAlunoProcessado.ScholarshipHolder = aluno.ScholarshipHolder;
        dadosAlunoProcessado.AgeAtEnrollment = aluno.AgeAtEnrollment;
        dadosAlunoProcessado.International = aluno.International;
        dadosAlunoProcessado.CurricularUnits1stSemCredited = aluno.CurricularUnits1stSemCredited;
        dadosAlunoProcessado.CurricularUnits1stSemEnrolled = aluno.CurricularUnits1stSemEnrolled;
        dadosAlunoProcessado.CurricularUnits1stSemEvaluations = aluno.CurricularUnits1stSemEvaluations;
        dadosAlunoProcessado.CurricularUnits1stSemApproved = aluno.CurricularUnits1stSemApproved;
        dadosAlunoProcessado.CurricularUnits1stSemGrade = aluno.CurricularUnits1stSemGrade;
        dadosAlunoProcessado.CurricularUnits1stSemWithoutEvaluations = aluno.CurricularUnits1stSemWithoutEvaluations;
        dadosAlunoProcessado.CurricularUnits2ndSemCredited = aluno.CurricularUnits2ndSemCredited;
        dadosAlunoProcessado.CurricularUnits2ndSemEnrolled = aluno.CurricularUnits2ndSemEnrolled;
        dadosAlunoProcessado.CurricularUnits2ndSemEvaluations = aluno.CurricularUnits2ndSemEvaluations;
        dadosAlunoProcessado.CurricularUnits2ndSemApproved = aluno.CurricularUnits2ndSemApproved;
        dadosAlunoProcessado.CurricularUnits2ndSemGrade = aluno.CurricularUnits2ndSemGrade;
        dadosAlunoProcessado.CurricularUnits2ndSemWithoutEvaluations = aluno.CurricularUnits2ndSemWithoutEvaluations;
        dadosAlunoProcessado.UnemploymentRate = aluno.UnemploymentRate;
        dadosAlunoProcessado.InflationRate = aluno.InflationRate;
        dadosAlunoProcessado.GDP = aluno.GDP;
        dadosAlunoProcessado.TargetBol = aluno.Target.Equals("dropout", StringComparison.CurrentCultureIgnoreCase);
    }


    /// <summary>
    /// Treina e avalia diversos modelos de classificação
    /// </summary>
    private (ITransformer Model, IDataView TrainingDataView) TreinarAvaliarModelos(IDataView dadosTreino, IDataView dadosTeste)
    {
        Console.WriteLine("\n=== Treinamento e Avaliação de Modelos ===");

        string[] features = [.. dadosTreino.Schema.Where(s => !s.Name.StartsWith("Target")).Select(s => s.Name)];

        var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", features);

        IDataView preprocessedTrainingData = preprocessingPipeline.Fit(dadosTreino).Transform(dadosTreino);

        // Lista de algoritmos a serem testados
        List<(string Name, IEstimator<ITransformer> Pipeline)> modelBuilders =
        [
                ("Regressão Logística",
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                        labelColumnName: "TargetBol", featureColumnName: "Features")),

                ("Regressão Logística SDCA",
                    mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                        labelColumnName: "TargetBol", featureColumnName: "Features")),

                ("Árvore de Decisão",
                    mlContext.BinaryClassification.Trainers.FastTree(
                        labelColumnName: "TargetBol", featureColumnName: "Features",
                        numberOfLeaves: 200, numberOfTrees: 600, minimumExampleCountPerLeaf: 50)),

                ("LightGBM",
                    mlContext.BinaryClassification.Trainers.LightGbm(
                        labelColumnName: "TargetBol", featureColumnName: "Features",
                        numberOfLeaves: 200, numberOfIterations: 600, minimumExampleCountPerLeaf: 50))
            ];

        // Armazenar resultados para comparação
        List<(string Name, ITransformer Model, BinaryClassificationMetrics Metrics)> modelResults = [];

        // Treinar e avaliar cada modelo
        foreach ((string name, IEstimator<ITransformer> estimator) in modelBuilders)
        {
            Console.WriteLine($"\nTreinando modelo: {name}");

            // Treina o modelo
            ITransformer trainedModel = estimator.Fit(preprocessedTrainingData);

            // Aplicar o modelo aos dados de teste
            IDataView testDataTransformed = preprocessingPipeline.Fit(dadosTeste).Transform(dadosTeste);
            IDataView predictions = trainedModel.Transform(testDataTransformed);

            // Avaliar o modelo
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "TargetBol",
                scoreColumnName: "Score",
                probabilityColumnName: "Probability");

            // Exibir métricas
            Console.WriteLine($"Acurácia: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"Precisão: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall:P2}");

            // Adicionar aos resultados
            modelResults.Add((name, trainedModel, metrics));
        }

        // Identificar o melhor modelo (usando AUC como métrica principal)
        (string Name, ITransformer Model, BinaryClassificationMetrics Metrics) bestModel = modelResults.OrderByDescending(m => m.Metrics.AreaUnderRocCurve).First();

        Console.WriteLine($"\nMelhor modelo: {bestModel.Name} com AUC = {bestModel.Metrics.AreaUnderRocCurve:P2}");

        // Retorna o modelo completo (preprocessamento + melhor algoritmo)
        EstimatorChain<ITransformer> bestPipeline = preprocessingPipeline.Append(
            modelBuilders.First(m => m.Name == bestModel.Name).Pipeline);

        TransformerChain<ITransformer> finalModel = bestPipeline.Fit(dadosTreino);

        return (finalModel, dadosTreino);
    }
}
