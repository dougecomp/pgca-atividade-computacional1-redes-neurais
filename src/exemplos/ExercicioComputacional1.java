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
import org.neuroph.core.learning.error.MeanSquaredError;
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

    private static NeuralNetworkLearningEventListener listenerTreinamento;
    /**
     * Método para leitura de um arquivo em txt. Poderá ser utilizado para os três conjuntos (treino,validacao,teste)
     * @param filename
     * @param separator
     * @return
     * @throws IOException 
     */
    public static DataSet getDataSet(String filename, String separator) throws IOException {
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

        DataSet dataSet = new DataSet(1, 1);
        // Ler o conjunto de dados de um arquivo
        
        String line = reader.readLine();
        while(line!=null){
            double[] x = new double[1];
            double[] y = new double[1];
            String[] partes = line.split(separator);
            
            x[0] = Double.parseDouble(partes[0]);
            if(x[0] > 1) {
                x[0] = 0.99;
            } else if(x[0] < -1) {
                x[0] = -0.99;
            } 
            
            y[0] = Double.parseDouble(partes[1]);
            if(y[0] > 1) {
                y[0] = 0.99;
            } else if(y[0] < -1) {
                y[0] = -0.99;
            }

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
    public static MultiLayerPerceptron createNnet(int qtdInputNeurons, int qtdHiddenNeurons, int qtdOutputNeurons) {
        
        // Armazenando os dados da quantidade de neurônios em cada camada
        ArrayList<Integer> neuronsInLayers = new ArrayList<>();
        neuronsInLayers.add(qtdInputNeurons); // Primeira camada: 1 ( X )
        neuronsInLayers.add(qtdHiddenNeurons); // Segunda camada: 5 ou 20
        neuronsInLayers.add(qtdOutputNeurons); // Terceira camada: 1 ( Y )
        
        // Instanciando rede neural. Verificar se a função de transferência é a sigmoide mesmo pois o intervalo indicado no exercício é de -1 a 1.
        // A sigmoid é de 0 a 1. A hiperbólica vai de -1 a 1 como o exercício diz.
        NeuronProperties np = new NeuronProperties(TransferFunctionType.TANH, true); // Configurando função de transferência e explicitando que os neurônios tem bias
        
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(neuronsInLayers, np);
        
        return mlp;
        
    }
    
    public static void trainNnet(NeuralNetwork nnet, DataSet data, double learningRate, double momentum) {
        
        //Configurando função de Erro Quadrático Médio
        MeanSquaredError mse = new MeanSquaredError();
        mse.reset();
        int maxIterations = 200;
        
        // Configurando tipo de treinamento
        if(momentum > 0) {
            MomentumBackpropagation lr = new MomentumBackpropagation();
            lr.setMomentum(momentum);
            lr.setErrorFunction(mse);
            lr.setMaxIterations(maxIterations);
            lr.setLearningRate(learningRate);
            nnet.setLearningRule(lr);
        } else {
            BackPropagation lr = new BackPropagation();
            lr.setErrorFunction(mse);
            lr.setMaxIterations(maxIterations);
            lr.setLearningRate(learningRate);
            nnet.setLearningRule(lr);
        }
        
        nnet.randomizeWeights(-0.1, 0.1); // Gerar pesos aleatório entre -0.1 e 0.1 antes de treinar
        
        //NeuralNetworkValidationListener nnvl = new NeuralNetworkValidationListener();
        NeuralNetworkLearningEventListener nnlel = new NeuralNetworkLearningEventListener();
        nnet.getLearningRule().addListener(nnlel);
        nnet.learn(data);
        listenerTreinamento = nnlel;
        System.out.println("Menor Erro: "+nnlel.menorErro);
        System.out.println("Vetor de pesos com menor erro: ("+nnlel.pesosMenorErro.length+" pesos) "+Arrays.toString(nnlel.pesosMenorErro));
        plotarGrafico(nnlel.erros, "Erro Quadrático Médio Durante Treinamento", "Épocas", "Erro");
        nnet.getLearningRule().removeListener(nnlel);
        
    }
    
    public static void validateNnet(NeuralNetwork nnet, DataSet validationSet, double learningRate, double momentum) {
        NeuralNetworkValidationListener nnvl = new NeuralNetworkValidationListener();
        nnet.getLearningRule().addListener(nnvl);
        
        // Convertendo de double[] para Double[]
        // Isso foi feito devido o getWeights retornar Double[] e o setWeigths receber double[]
        double[] pesos = new double[]{};
        for (int i = 0; i < listenerTreinamento.pesosMenorErro.length; i++) {
            pesos[i] = listenerTreinamento.pesosMenorErro[i];
        }
        nnet.setWeights(pesos);

        //Validar a rede com os dados de validação e os vetor de pesos com o menor erro durante o treinamento.
        trainNnet(nnet, validationSet, learningRate, momentum);
        nnet.getLearningRule().removeListener(nnvl);
    }
    
    public static void testNnet(NeuralNetwork nnet, DataSet testSet) {
        XYSeries pontos = new XYSeries("Função");
        for(DataSetRow row : testSet.getRows()) {
            nnet.setInput(row.getInput());
            nnet.calculate();
            double[] networkOutput = nnet.getOutput();
            pontos.add(row.getInput()[0], networkOutput[0]);
            System.out.print("Input: " + Arrays.toString(row.getInput()) );
            System.out.println("Output: " + Arrays.toString(networkOutput) );
        }
        plotarGrafico(pontos, "Função Aproximada Através da Rede Neural", "X", "Y");
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
        double learningRate = 0.01;
        double momentum = 0;
        String filenameTraining = "dataSetTraining.txt";
        String filenameValidation = "dataSetValidation.txt";
        String filenameTest = "dataSetTest.txt";
        String separator = ";";
        
        //Criando rede neural
        MultiLayerPerceptron mlp = createNnet(1, 5, 1);
        
        //Lendo conjunto de treinamento
        DataSet ds = getDataSet(filenameTraining, separator);
        
        //Treinando a rede neural
        trainNnet(mlp, ds, learningRate, momentum);
        
        //Lendo conjunto de validação
        ds = getDataSet(filenameValidation, separator);
        
        //Validando a rede neural
        validateNnet(mlp, ds, learningRate, momentum);
        
        //Lendo conjunto de teste
        ds = getDataSet(filenameTest, separator);
        
        //Testando rede neural
        testNnet(mlp, ds);
    }
    
}
