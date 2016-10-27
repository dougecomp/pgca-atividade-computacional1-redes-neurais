/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package exemplos;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import javax.swing.JFrame;
import listeners.NeuralNetworkLearningEventListener;
import listeners.NeuralNetworkValidationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author douglas
 */
public class ExercicioComputacional1 {

    /**
     * Método para leitura de um arquivo em txt. Poderá ser utilizado para os três conjuntos (treino,validacao,teste)
     * @param filename
     * @param separator
     * @return
     * @throws IOException 
     */
    public static DataSet getDataSet(String filename, String separator) throws IOException {
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
                                    
        double[] x = new double[1];
        double[] y = new double[1];

        DataSet dataSet = new DataSet(1, 1);
        // Ler o conjunto de dados de um arquivo
        
        String line = reader.readLine();
        while(line!=null){
            String[] partes = line.split(separator);
            
            x[0] = Double.parseDouble(partes[0]);
            
            y[0] = Double.parseDouble(partes[1]);

            dataSet.addRow(x, y);

            line = reader.readLine();
        }   
        reader.close();
        
        return dataSet;
        
    }
    
    /**
     * Método para criação de uma rede neural com três camadas (entrada, escondida e saída).
     * No caso desse trabalho qtdInputNeurons e qtdOutputNeuros será sempre 1. Variar somente qtdHiddenNeurons de 5 para 20.
     * @param qtdInputNeurons 
     * @param qtdHiddenNeurons  
     * @param qtdOutputNeurons
     * @return 
     */
    public static NeuralNetwork createNnet(int qtdInputNeurons, int qtdHiddenNeurons, int qtdOutputNeurons) {
        
        // Armazenando os dados da quantidade de neurônios em cada camada
        ArrayList<Integer> neuronsInLayers = new ArrayList<>();
        neuronsInLayers.add(qtdInputNeurons); // Primeira camada: 1 ( X )
        neuronsInLayers.add(qtdHiddenNeurons); // Segunda camada: 5 ou 20
        neuronsInLayers.add(qtdOutputNeurons); // Terceira camada: 1 ( Y )
        
        // Instanciando rede neural. Verificar se a função de transferência é a sigmoide mesmo pois o intervalo indicado no exercício é de -1 a 1.
        // A sigmoid é de 0 a 1. A hiperbólica vai de -1 a 1 como o exercício diz.
        NeuronProperties np = new NeuronProperties(TransferFunctionType.TANH, true); // Configurando função de transferência e explicitando que os neurônios tem bias
        
        NeuralNetwork mlp = new MultiLayerPerceptron(neuronsInLayers, np);
        
        return mlp;
        
    }
    
    public static void trainNnet(NeuralNetwork nnet, DataSet data, double learningRate, double momentum) {
        
        // Configurando tipo de treinamento
        if(momentum > 0) {
            MomentumBackpropagation lr = new MomentumBackpropagation();
            lr.setMomentum(momentum);
            nnet.setLearningRule(lr);
        } else {
            BackPropagation lr = new BackPropagation();
            nnet.setLearningRule(lr);
        }
        
        // Configurando taxa de aprendizado
        if(nnet.getLearningRule() instanceof MomentumBackpropagation) {
            ((MomentumBackpropagation) nnet.getLearningRule()).setLearningRate(learningRate);
        } else if(nnet.getLearningRule() instanceof BackPropagation) {
            ((BackPropagation) nnet.getLearningRule()).setLearningRate(learningRate);
        }
        
        nnet.randomizeWeights(-0.5, 0.5); // Gerar pesos aleatório entre -0.5 e 0.5 antes de treinar
        
        nnet.learn(data);
        
    }
    
    public static void validateNnet(NeuralNetwork nnet, DataSet validationSet, double learningRate, double momentum) {
        NeuralNetworkValidationListener nnvl = new NeuralNetworkValidationListener();
        nnet.addListener(nnvl);
        //testNnet(nnet, validationSet);
        trainNnet(nnet, validationSet, learningRate, momentum);
        nnet.removeListener(nnvl);
    }
    
    public static void testNnet(NeuralNetwork nnet, DataSet testSet) {
        for(DataSetRow row : testSet.getRows()) {
            nnet.setInput(row.getInput());
            nnet.calculate();
            double[] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(row.getInput()) );
            System.out.println("Output: " + Arrays.toString(networkOutput) );
        }
    }
    
    public static void plotarGrafico(XYSeries pontos, String tituloGrafico, String nomeEixoX, String nomeEixoY) {
        

        XYSeriesCollection dados = new XYSeriesCollection();
        dados.addSeries(pontos);

        JFreeChart grafico = ChartFactory.createXYLineChart(
            tituloGrafico,
            nomeEixoX,
            nomeEixoY,
            dados, PlotOrientation.VERTICAL, true, true, true
        );

        ChartPanel panel = new ChartPanel(grafico);
        JFrame frame = new JFrame();
        frame.setSize(640, 480);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(panel);
        frame.setVisible(true);
		
        
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        double learningRate = 0.5;
        double momentum = 0;
        
        // Gráfico da época (X) pelo erro quadrático (Y)
        XYSeries erros = new XYSeries("Erro da Rede");
    }
    
}
