using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations.Schema;
using System.Reflection.Emit;

namespace PrevisaoEvasaoEscolar.Models;

/// <summary>
/// Classe que representa os dados de entrada do aluno
/// </summary>
public record DadosAluno
{
    [LoadColumn(0)]
    public float MaritalStatus { get; set; }

    [LoadColumn(1)]
    public float ApplicationMode { get; set; }

    [LoadColumn(2)]
    public float ApplicationOrder { get; set; }

    [LoadColumn(3)]
    public float Course { get; set; }

    [LoadColumn(4)]
    public float DaytimeEveningAttendance { get; set; }

    [LoadColumn(5)]
    public float PreviousQualification { get; set; }

    [LoadColumn(6)]
    public float Nacionality { get; set; }

    [LoadColumn(7)]
    public float MothersQualification { get; set; }

    [LoadColumn(8)]
    public float FathersQualification { get; set; }

    [LoadColumn(9)]
    public float MothersOccupation { get; set; }

    [LoadColumn(10)]
    public float FathersOccupation { get; set; }

    [LoadColumn(11)]
    public float Displaced { get; set; }

    [LoadColumn(12)]
    public float EducationalSpecialNeeds { get; set; }

    [LoadColumn(13)]
    public float Debtor { get; set; }

    [LoadColumn(14)]
    public float TuitionFeesUpToDate { get; set; }

    [LoadColumn(15)]
    public float Gender { get; set; }

    [LoadColumn(16)]
    public float ScholarshipHolder { get; set; }

    [LoadColumn(17)]
    public float AgeAtEnrollment { get; set; }

    [LoadColumn(18)]
    public float International { get; set; }

    [LoadColumn(19)]
    public float CurricularUnits1stSemCredited { get; set; }

    [LoadColumn(20)]
    public float CurricularUnits1stSemEnrolled { get; set; }

    [LoadColumn(21)]
    public float CurricularUnits1stSemEvaluations { get; set; }

    [LoadColumn(22)]
    public float CurricularUnits1stSemApproved { get; set; }

    [LoadColumn(23)]
    public float CurricularUnits1stSemGrade { get; set; }

    [LoadColumn(24)]
    public float CurricularUnits1stSemWithoutEvaluations { get; set; }

    [LoadColumn(25)]
    public float CurricularUnits2ndSemCredited { get; set; }

    [LoadColumn(26)]
    public float CurricularUnits2ndSemEnrolled { get; set; }

    [LoadColumn(27)]
    public float CurricularUnits2ndSemEvaluations { get; set; }

    [LoadColumn(28)]
    public float CurricularUnits2ndSemApproved { get; set; }

    [LoadColumn(29)]
    public float CurricularUnits2ndSemGrade { get; set; }

    [LoadColumn(30)]
    public float CurricularUnits2ndSemWithoutEvaluations { get; set; }

    [LoadColumn(31)]
    public float UnemploymentRate { get; set; }

    [LoadColumn(32)]
    public float InflationRate { get; set; }

    [LoadColumn(33)]
    public float GDP { get; set; }

    [LoadColumn(34)]
    public string Target { get; set; }
}

public record DadosAlunoProcessado
{
    public float MaritalStatus { get; set; }

    public float ApplicationMode { get; set; }

    public float ApplicationOrder { get; set; }

    public float Course { get; set; }

    public float DaytimeEveningAttendance { get; set; }

    public float PreviousQualification { get; set; }

    public float Nacionality { get; set; }

    public float MothersQualification { get; set; }

    public float FathersQualification { get; set; }

    public float MothersOccupation { get; set; }

    public float FathersOccupation { get; set; }

    public float Displaced { get; set; }

    public float EducationalSpecialNeeds { get; set; }

    public float Debtor { get; set; }

    public float TuitionFeesUpToDate { get; set; }

    public float Gender { get; set; }

    public float ScholarshipHolder { get; set; }

    public float AgeAtEnrollment { get; set; }

    public float International { get; set; }

    public float CurricularUnits1stSemCredited { get; set; }

    public float CurricularUnits1stSemEnrolled { get; set; }

    public float CurricularUnits1stSemEvaluations { get; set; }

    public float CurricularUnits1stSemApproved { get; set; }

    public float CurricularUnits1stSemGrade { get; set; }

    public float CurricularUnits1stSemWithoutEvaluations { get; set; }

    public float CurricularUnits2ndSemCredited { get; set; }

    public float CurricularUnits2ndSemEnrolled { get; set; }

    public float CurricularUnits2ndSemEvaluations { get; set; }

    public float CurricularUnits2ndSemApproved { get; set; }

    public float CurricularUnits2ndSemGrade { get; set; }

    public float CurricularUnits2ndSemWithoutEvaluations { get; set; }

    public float UnemploymentRate { get; set; }

    public float InflationRate { get; set; }

    public float GDP { get; set; }
    public bool TargetBol { get; set; }
}
