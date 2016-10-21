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
import listeners.NeuralNetworkValidationListener;
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

    public static DataSet getDataSet(String filename, String separator) throws IOException {
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
                                    
        double[] x = new double[1];
        double[] y = new double[1];

        DataSet trainingSet = new DataSet(1, 1);
        // Ler o conjunto de dados de um arquivo
        
        String line = reader.readLine();
        while(line!=null){
            String[] partes = line.split(separator);
            
            x[0] = Double.parseDouble(partes[0]);
            
            y[0] = Double.parseDouble(partes[1]);

            trainingSet.addRow(x, y);

            line = reader.readLine();
        }   
        reader.close();
        
        return trainingSet;
        
    }
    
    public static NeuralNetwork createNnet() {
        
        int qtdInputNeurons = 1;
        int qtdHiddenNeuros = 5;
        int qtdOutputNeurons = 1;
        
        // Armazenando os dados da quantidade de neurônios em cada camada
        ArrayList<Integer> neuronsInLayers = new ArrayList<>();
        neuronsInLayers.add(qtdInputNeurons); // Primeira camada: 1 ( X )
        neuronsInLayers.add(qtdHiddenNeuros); // Segunda camada: 5 ou 20
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
        nnet.addListener(new NeuralNetworkValidationListener());
        //testNnet(nnet, validationSet);
        trainNnet(nnet, validationSet, learningRate, momentum);
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
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        double learningRate = 0.5;
        double momentum = 0;
    }
    
}
