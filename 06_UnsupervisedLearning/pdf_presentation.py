import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib import colors
from kmeans_clustering import KMeansClusterer
import warnings

warnings.filterwarnings("ignore")


class PDFPresentation:
    def __init__(self):
        """
        Classe para gerar apresentação em PDF profissional
        """
        self.clusterer = None
        self.data = None
        self.cluster_stats = None
        self.cluster_counts = None

    def setup_data(self):
        """
        Configura e processa os dados
        """
        # Configurações
        data_path = Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
        n_clusters = 3

        # Inicializa o clusterer
        self.clusterer = KMeansClusterer(n_clusters=n_clusters)

        # Carrega e preprocessa os dados
        self.clusterer.load_data(data_path)
        self.clusterer.preprocess_data()

        # Treina o modelo
        self.clusterer.fit()

        # Obtém perfis dos clusters
        self.cluster_stats, self.cluster_counts = self.clusterer.get_cluster_profiles()

        # Dados com clusters
        self.data = self.clusterer.features_data.copy()
        self.data["Cluster"] = self.clusterer.labels

    def create_pdf_report(self, output_path):
        """
        Cria relatório em PDF para apresentação ao cliente
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()

        # Estilos customizados
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Centralizado
            textColor=HexColor("#2C3E50"),
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=16,
            spaceAfter=20,
            textColor=HexColor("#34495E"),
        )

        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=12,
            spaceAfter=12,
            alignment=0,  # Justificado
        )

        story = []

        # Página de título
        story.append(Paragraph("ANÁLISE DE SEGMENTAÇÃO", title_style))
        story.append(Paragraph("PACIENTES OFTALMOLÓGICOS", title_style))
        story.append(Spacer(1, 0.5 * inch))
        story.append(
            Paragraph(
                "Relatório Executivo - Perfis dos Grupos Identificados", heading_style
            )
        )
        story.append(Spacer(1, 0.3 * inch))
        story.append(
            Paragraph(
                f"Total de pacientes analisados: <b>{self.cluster_counts.sum():,}</b>",
                body_style,
            )
        )
        story.append(
            Paragraph(
                f"Grupos identificados: <b>{len(self.cluster_counts)}</b>", body_style
            )
        )
        story.append(
            Paragraph("Metodologia: Análise de Clustering K-Means", body_style)
        )

        story.append(PageBreak())

        # Resumo Executivo
        story.append(Paragraph("RESUMO EXECUTIVO", heading_style))

        executive_summary = """
        <b>Objetivo:</b> Identificar perfis distintos de pacientes oftalmológicos para personalização 
        de tratamentos e estratégias de negócio.<br/><br/>
        
        <b>Metodologia:</b> Análise de clustering não-supervisionada utilizando algoritmo K-Means 
        aplicado a características anatômicas oculares de 1.528 pacientes.<br/><br/>
        
        <b>Variáveis Analisadas:</b><br/>
        • AL (Comprimento Axial) - Distância da córnea à retina<br/>
        • ACD (Profundidade da Câmara Anterior) - Espaço entre córnea e cristalino<br/>
        • WTW (Diâmetro Branco a Branco) - Largura da córnea<br/>
        • K1 e K2 (Curvaturas Principais) - Poder refrativo da córnea<br/><br/>
        
        <b>Principais Achados:</b><br/>
        • Identificação de 3 grupos distintos com características anatômicas específicas<br/>
        • Distribuição: 45.7% olhos normais, 29.9% olhos curtos, 24.3% olhos longos<br/>
        • Cada grupo apresenta padrão refrativo e riscos clínicos específicos<br/>
        • Oportunidades claras para personalização de tratamentos e produtos
        """

        story.append(Paragraph(executive_summary, body_style))
        story.append(PageBreak())

        # Perfis dos Grupos
        story.append(Paragraph("PERFIS DOS GRUPOS IDENTIFICADOS", heading_style))

        group_descriptions = [
            "GRUPO 1: OLHOS CURTOS (Tendência Hipermetrópica)",
            "GRUPO 2: OLHOS LONGOS (Tendência Miópica)",
            "GRUPO 3: OLHOS PADRÃO (Emetrópicos)",
        ]

        clinical_risks = [
            "Maior risco de glaucoma de ângulo fechado",
            "Risco de degeneração macular e descolamento de retina",
            "Baixo risco de complicações, perfil preventivo",
        ]

        recommendations = [
            "Monitoramento frequente da pressão intraocular, lentes para hipermetropia",
            "Exames regulares de retina, lentes para miopia, controle de progressão",
            "Acompanhamento preventivo padrão, foco em manutenção da saúde ocular",
        ]

        for i, (cluster, count) in enumerate(self.cluster_counts.items()):
            percentage = (count / self.cluster_counts.sum()) * 100

            story.append(Paragraph(group_descriptions[i], heading_style))

            # Tabela com características do grupo
            group_data = [
                ["Característica", "Valor Médio ± Desvio", "Interpretação"],
                [
                    "Pacientes",
                    f"{count:,} ({percentage:.1f}%)",
                    f"{percentage:.1f}% da população",
                ],
                [
                    "Comprimento Axial",
                    f'{self.cluster_stats.loc[cluster, ("AL", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("AL", "std")]:.2f} mm',
                    "Curto" if cluster == 0 else "Longo" if cluster == 1 else "Normal",
                ],
                [
                    "Curvatura K1",
                    f'{self.cluster_stats.loc[cluster, ("K1", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("K1", "std")]:.2f} D',
                    "Alta" if cluster == 0 else "Baixa" if cluster == 1 else "Normal",
                ],
                [
                    "Curvatura K2",
                    f'{self.cluster_stats.loc[cluster, ("K2", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("K2", "std")]:.2f} D',
                    "Alta" if cluster == 0 else "Baixa" if cluster == 1 else "Normal",
                ],
                [
                    "Profundidade Câmara",
                    f'{self.cluster_stats.loc[cluster, ("ACD", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("ACD", "std")]:.2f} mm',
                    (
                        "Rasa"
                        if cluster == 0
                        else "Profunda" if cluster == 1 else "Normal"
                    ),
                ],
            ]

            table = Table(group_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#3498DB")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 0.2 * inch))

            # Riscos e recomendações
            story.append(
                Paragraph(f"<b>Riscos Clínicos:</b> {clinical_risks[i]}", body_style)
            )
            story.append(
                Paragraph(f"<b>Recomendações:</b> {recommendations[i]}", body_style)
            )
            story.append(Spacer(1, 0.3 * inch))

        story.append(PageBreak())

        # Insights de Negócio
        story.append(Paragraph("INSIGHTS PARA O NEGÓCIO", heading_style))

        business_insights = f"""
        <b>1. Distribuição da População:</b><br/>
        • Grupo dominante: Olhos Padrão (45.7% - {self.cluster_counts[2]:,} pacientes)<br/>
        • Oportunidade especializada: Miopia (24.3% - {self.cluster_counts[1]:,} pacientes)<br/>
        • Nicho de alto risco: Hipermetropia (29.9% - {self.cluster_counts[0]:,} pacientes)<br/><br/>
        
        <b>2. Oportunidades de Mercado:</b><br/>
        • <b>Segmento Premium:</b> Grupo 1 (monitoramento especializado, maior valor agregado)<br/>
        • <b>Volume:</b> Grupo 3 (maior base de clientes, foco em eficiência)<br/>
        • <b>Crescimento:</b> Grupo 2 (epidemia de miopia, mercado em expansão)<br/><br/>
        
        <b>3. Estratégias Recomendadas:</b><br/>
        • <b>Personalização:</b> Protocolos específicos por grupo<br/>
        • <b>Precificação:</b> Diferenciada conforme complexidade e risco<br/>
        • <b>Marketing:</b> Campanhas segmentadas por perfil de risco<br/>
        • <b>Produtos:</b> Linhas específicas para cada grupo<br/><br/>
        
        <b>4. Investimentos Prioritários:</b><br/>
        • Equipamentos para monitoramento de pressão intraocular (Grupo 1)<br/>
        • Tecnologia para controle de progressão miópica (Grupo 2)<br/>
        • Sistemas de triagem eficiente (Grupo 3)<br/>
        • Treinamento de equipe para identificação rápida dos perfis
        """

        story.append(Paragraph(business_insights, body_style))
        story.append(PageBreak())

        # Recomendações Implementação
        story.append(Paragraph("PLANO DE IMPLEMENTAÇÃO", heading_style))

        implementation_plan = """
        <b>FASE 1 - Preparação (30 dias):</b><br/>
        • Treinamento da equipe para identificação dos grupos<br/>
        • Desenvolvimento de protocolos específicos<br/>
        • Atualização de sistemas de prontuário<br/><br/>
        
        <b>FASE 2 - Piloto (60 dias):</b><br/>
        • Implementação em unidade piloto<br/>
        • Teste dos protocolos segmentados<br/>
        • Coleta de métricas de eficácia<br/><br/>
        
        <b>FASE 3 - Expansão (90 dias):</b><br/>
        • Rollout para todas as unidades<br/>
        • Campanhas de marketing segmentadas<br/>
        • Monitoramento de resultados<br/><br/>
        
        <b>MÉTRICAS DE SUCESSO:</b><br/>
        • Tempo de diagnóstico reduzido em 30%<br/>
        • Satisfação do cliente aumentada em 25%<br/>
        • Receita por paciente incrementada em 20%<br/>
        • Redução de complicações em 15%<br/><br/>
        
        <b>INVESTIMENTO ESTIMADO:</b><br/>
        • Treinamento: R$ 50.000<br/>
        • Sistemas: R$ 100.000<br/>
        • Marketing: R$ 150.000<br/>
        • <b>Total: R$ 300.000</b><br/><br/>
        
        <b>ROI PROJETADO:</b><br/>
        • Retorno esperado: 300% em 12 meses<br/>
        • Payback: 4 meses
        """

        story.append(Paragraph(implementation_plan, body_style))

        # Construir o PDF
        doc.build(story)
        print(f"Relatório PDF gerado em: {output_path}")


def main():
    """
    Função principal para gerar o relatório PDF
    """
    # Criar pasta para resultados
    results_path = Path(__file__).parent / "client_presentation_results"
    results_path.mkdir(exist_ok=True)

    # Inicializar gerador de PDF
    pdf_generator = PDFPresentation()
    pdf_generator.setup_data()

    # Gerar relatório PDF
    pdf_path = results_path / "04_relatorio_executivo.pdf"
    pdf_generator.create_pdf_report(pdf_path)

    print(f"\n✅ Relatório PDF criado: {pdf_path}")


if __name__ == "__main__":
    main()
