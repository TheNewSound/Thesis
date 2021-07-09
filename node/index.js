const argparse = require('argparse');
const fs = require('fs');
let fe = require('./featureExtractors.js');
let sm = require('./similarityMetrics.js');
let ca = require('./clusteringAlgorithms.js');
let tf;


function getLeafNodes(nodes, result = []) {
    for (let i = 0; i < nodes.length; i++) {
        if (!Array.isArray(nodes[i][0])) {
            result.push(nodes[i]);
        } else {
            result = getLeafNodes(nodes[i], result);
        }
    }
    return result;
}

function tensor2DToImage(matrix, filename)
{
    const file = './output/'+filename;
    return tf.tidy(() => {
        tf.node.encodeJpeg(matrix.mul(255).cast('int32').expandDims(2), 'grayscale', 100).then((data) => {
            /*fs.access(file, fs.constants.W_OK, function(err){
                if(!err){*/
                    fs.writeFile(file, Buffer.from(data), function (err) { });
                /*}
            });*/
        });
    });
}

function arrayToFile(array, filename)
{
    const file = './output/'+filename;
    fs.writeFile(file, JSON.stringify(array),
        function (err) {
            if (err) {
                console.error('Cannot write array to file');
            }
        }
    );
}

async function readValidation(csvUrl){
    const columns = await loadColumns(csvUrl);
    console.log(columns);
    if(columns.includes('Group')) {
        const csvDataset = tf.data.csv(
            'file://' + csvUrl, {
                hasHeader: true,
                configuredColumnsOnly: true, // important
                columnConfigs: {Group: {dtype: 'int32'}},
                delimiter: ';',
            });
        let groups = [];
        const flattened = await csvDataset.forEachAsync(x => {
            groups.push(x.Group);
        });
        groups = groups.map((x, i) => (x<0)?x-i:x+=1); //Fix groups with ID 0 and -1 values (noise)
        console.log(groups)
        const max = Math.max(...groups);
        return tf.tidy(() => {
            const groupsTensor = tf.tensor1d(groups);
            const matrix = tf.outerProduct(groupsTensor, groupsTensor).sub(groupsTensor.square()).equal(0);
            return matrix;
        });
    }
    return null;
}

async function loadColumns(csvUrl){
    let lineReader = require('readline').createInterface({
        input: fs.createReadStream(csvUrl),
    });
    return new Promise(function(resolve, reject) {
        lineReader.on('line', function (line) {
            this.close();
            resolve(line.split(';'));
        });
    });
}

async function read(csvUrl, csvConfig) {
    let timer = process.hrtime();
    const csvDataset = tf.data.csv(
        'file://' + csvUrl, {
            hasHeader: true,
            configuredColumnsOnly: true, // important
            columnConfigs: csvConfig,
            delimiter: ';',
        });
    let d = process.hrtime(timer)
    console.info("Reading data: %ds %dms", d[0], d[1] / 1000000);
    timer = process.hrtime();
    let features = {};
    let extractedFeatures = [];
    (await csvDataset.columnNames()).forEach(n => {
        extractedFeatures.push([]);
        features[n] = [];
    });
    await csvDataset.forEachAsync(x => {
        for (let [key, value] of Object.entries(x)) {
            features[key].push(value);
        }
    });
    //console.log(features["Date"][2879]);
    // MERGE Name + Description into Description
    //features["Description"].map((x,i) => x + features["Name"][i]);
    let j = 0;
    for (let [key, value] of Object.entries(features)) {
        for (let i = 0; i < csvConfig[key].fextractors.length; ++i) {
            extractedFeatures[j][i] = csvConfig[key].fextractors[i](value); // run the extractor
        }
        j++;
    }
    //console.log(extractedFeatures[3]);
    d = process.hrtime(timer);
    console.info("Extracting features: %ds %dms", d[0], d[1] / 1000000);
    timer = process.hrtime();
    const flatFeatures = getLeafNodes(extractedFeatures).map(i => {
        if (Array.isArray(i)) {
            if (i.length == 1) {
                return i[0];
            }
            return tf.tensor(i);
        }
        return i;
    });
    d = process.hrtime(timer);
    console.info("Flattening features: %ds %dms", d[0], d[1] / 1000000);
    return flatFeatures;
}

function logTfInfo(){
    const stats = tf.memory();
    console.log('# unique tensors:'+stats.numTensors+", # allocated buffers:"+stats.numDataBuffers);
}

async function runPerceptron(csvUrl, csvConfig, args) {
    const validation = await readValidation(csvUrl);
    logTfInfo();
    const extractedFeatures = await read(csvUrl, csvConfig);
    if(args.gpu) {
        // test so cublas will be loaded before we start timing
        extractedFeatures.forEach(tensor => {
            tensor.print();
        })
        sm.VH(extractedFeatures[0], {removeCommon: true}).dispose();
    }
    //console.log(extractedFeatures[8]);
    //extractedFeatures[8].print()
    let timer = process.hrtime();
    let similarityFeatures = extractedFeatures.map((tensor,i) => {
        let ret;
        if(args.metric=="DICEsquared"){
            ret = sm.VH(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="DICE"){
            ret = sm.DICE(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="COSINE"){
            ret = sm.cosine(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="KUMAR"){
            ret = sm.kumar(tensor, {removeCommon:(i!==8)});
        }
        tensor.dispose();
        return ret;
    });
    let d = process.hrtime(timer);
    //similarityFeatures.forEach((tensor,i)=>tensor2DToImage(similarityFeatures[i], 'SimilarityMatrix_'+i+'_ING_threshold_0_0.jpeg'));
    //let buffer = Buffer.from(similarityFeatures[7].dataSync().buffer);
    //fs.writeFileSync("output/BigramVectors_ING.bin", buffer);
    logTfInfo();
    const numFeatures = similarityFeatures.length;
    const width = similarityFeatures[0].shape[0];
    console.info("Calculating all similarities: %ds %dms", d[0], d[1] / 1000000);
    console.info("Number of features extracted: "+similarityFeatures.length);
    //let weights = Array(similarityFeatures.length).fill(1/similarityFeatures.length, 0 , similarityFeatures.length); // tf.variable(); for training??

    const inputs = tf.stack(similarityFeatures,2).reshape([width*width,numFeatures]);
    logTfInfo();

    let multilayer=false;
    if(args.task=="TRAIN_NN"){
        multilayer=true;
    }
    const model = createModel(inputs, multilayer);
    model.summary();
    await trainModel(model, inputs, validation.flatten(), args.epochs);
    console.log('Done Training');
    await model.save('file://output/model_'+args.append);
    /****/
}

async function evalPerceptron(csvUrl, csvConfig, args) {
    //const validation = await readValidation(csvUrl);
    const extractedFeatures = await read(csvUrl, csvConfig);
    if(args.gpu) {
        // test so cublas will be loaded before we start timing
        extractedFeatures.forEach(tensor => {
            tensor.print();
        })
        sm.VH(extractedFeatures[0], {removeCommon: true}).dispose();
    }
    //console.log(extractedFeatures[8]);
    //extractedFeatures[8].print()
    let timer = process.hrtime();
    let similarityFeatures = extractedFeatures.map((tensor,i) => {
        let ret;
        if(args.metric=="DICEsquared"){
            ret = sm.VH(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="DICE"){
            ret = sm.DICE(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="COSINE"){
            ret = sm.cosine(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="KUMAR"){
            ret = sm.kumar(tensor, {removeCommon:(i!==8)});
        }
        tensor.dispose();
        return ret;
    });
    let d = process.hrtime(timer);
    //similarityFeatures.forEach((tensor,i)=>tensor2DToImage(similarityFeatures[i], 'SimilarityMatrix_'+i+'_ING_threshold_0_0.jpeg'));
    //let buffer = Buffer.from(similarityFeatures[7].dataSync().buffer);
    //fs.writeFileSync("output/BigramVectors_ING.bin", buffer);
    logTfInfo();
    const numFeatures = similarityFeatures.length;
    console.info("Calculating all similarities: %ds %dms", d[0], d[1] / 1000000);
    console.info("Number of features extracted: "+similarityFeatures.length);
    //let weights = Array(similarityFeatures.length).fill(1/similarityFeatures.length, 0 , similarityFeatures.length); // tf.variable(); for training??

    const inputs = tf.stack(similarityFeatures,2).reshape([similarityFeatures[0].shape[0]*similarityFeatures[0].shape[0],numFeatures]);
    logTfInfo();

    const model = await tf.loadLayersModel('file://output/model_'+args.append+'/model.json');//'+args.append);

    model.summary();

    const result = model.predict(inputs).round().reshape([similarityFeatures[0].shape[0],similarityFeatures[0].shape[0]])//.print();
    result.print();
    console.log(result);
    const result2 = await ca.simpleCluster2(result, 0.5, true);

    let file = fs.createWriteStream('./output/PERCEPTRONlabels_'+args.append+'.txt');
    file.on('error', function(err) { /* error handling */ });
    file.write('Group\n');
    result2.clusters.forEach(function(v) { file.write(v+'\n'); });
    file.end();
    result.dispose();
    result2.data.dispose();
}

async function runSimmatrix(csvUrl, csvConfig, args) {
    const validation = await readValidation(csvUrl);
    logTfInfo();
    const extractedFeatures = await read(csvUrl, csvConfig);
    if(args.gpu) {
        // test so cublas will be loaded before we start timing
        extractedFeatures.forEach(tensor => {
            tensor.print();
        })
        sm.VH(extractedFeatures[0], {removeCommon: true}).dispose();
    }
    //console.log(extractedFeatures[8]);
    //extractedFeatures[8].print()
    let timer = process.hrtime();
    let similarityFeatures = extractedFeatures.map((tensor,i) => {
        let ret;
        if(args.metric=="DICEsquared"){
            ret = sm.VH(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="DICE"){
            ret = sm.DICE(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="COSINE"){
            ret = sm.cosine(tensor, {removeCommon:(i!==8)});
        }else if(args.metric=="KUMAR"){
            ret = sm.kumar(tensor, {removeCommon:(i!==8)});
        }
        tensor.dispose();
        return ret;
    });
    let d = process.hrtime(timer);
    //similarityFeatures.forEach((tensor,i)=>tensor2DToImage(similarityFeatures[i], 'SimilarityMatrix_'+i+'_ING_threshold_0_0.jpeg'));
    //let buffer = Buffer.from(similarityFeatures[7].dataSync().buffer);
    //fs.writeFileSync("output/BigramVectors_ING.bin", buffer);
    logTfInfo();
    const numFeatures = similarityFeatures.length;
    console.info("Calculating all similarities: %ds %dms", d[0], d[1] / 1000000);
    console.info("Number of features extracted: "+similarityFeatures.length);
    //let weights = Array(similarityFeatures.length).fill(1/similarityFeatures.length, 0 , similarityFeatures.length); // tf.variable(); for training??
    //timer = process.hrtime();
    let weights = [0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.6,0.1];
    let weights2 = [0.3,0.6,0.1];
    similarityFeatures = similarityFeatures.map((tensor,i) => {
        const t = tensor.cast('float32').mul(weights2[i]);
        tensor.dispose();
        return t
    });
    let stackedFeatures = tf.stack(similarityFeatures).sum(0);//.maximum(0.75).sub(0.75).mul(4);//.greater(1000*weights[9]).cast('int32')
    console.log("Line 298");
    stackedFeatures.print();

    tensor2DToImage(stackedFeatures, 'SimilarityMatrix_'+args.append+'.jpeg');
    buffer = Buffer.from(stackedFeatures.dataSync().buffer);
    fs.writeFileSync("output/SimilarityMatrix_"+args.append+".bin", buffer);
}

async function runGenerateImageResult(csvUrl, csvConfig, args){
    let validation = null;
    if (fs.existsSync(csvUrl)) {
        validation = await readValidation(csvUrl);
    }

    let csvUrl2 = '../python/output/HDBSCANlabels_'+args.append+'.txt';
    let results;
    if (fs.existsSync(csvUrl2)) {
        results = await readValidation(csvUrl2);
        tensor2DToImage(results, 'HDBSCANlabels_' + args.append + '.jpeg');
        if (validation) {
            let accuracy = ca.calculateAccuracy(results, validation);
            console.log("HDBSCAN Calculated accuracy: " + accuracy);
        }
        results.dispose();
    }

    csvUrl2 = '../python/output/HACSINGLElabels_' + args.append + '.txt';
    if (fs.existsSync(csvUrl2)) {
        results = await readValidation(csvUrl2);
        tensor2DToImage(results, 'HACSINGLElabels_' + args.append + '.jpeg');
        if (validation) {
            accuracy = ca.calculateAccuracy(results, validation);
            console.log("HACSINGLE Calculated accuracy: " + accuracy);
        }
        results.dispose();
    }

    csvUrl2 = '../python/output/HACCOMPLETElabels_' + args.append + '.txt';
    if (fs.existsSync(csvUrl2)) {
        results = await readValidation(csvUrl2);
        tensor2DToImage(results, 'HACCOMPLETElabels_' + args.append + '.jpeg');
        if (validation) {
            accuracy = ca.calculateAccuracy(results, validation);
            console.log("HACCOMPLETE Calculated accuracy: " + accuracy);
        }
        results.dispose();
    }

    csvUrl2 = './output/PERCEPTRONlabels_' + args.append + '.txt';
    if (fs.existsSync(csvUrl2)) {
        results = await readValidation(csvUrl2);
        tensor2DToImage(results, 'PERCEPTRONlabels_' + args.append + '.jpeg');
        if (validation) {
            accuracy = ca.calculateAccuracy(results, validation);
            console.log("PERCEPTRON Calculated accuracy: " + accuracy);
        }
        results.dispose();
    }


    /*results = await readValidation('./data/example2_puntcomma_delimited.csv');
    tensor2DToImage(results, 'GROUNDlabels_ING.jpeg');
    accuracy=ca.calculateAccuracy(results,validation);
    console.log("GROUND TRUTH Calculated accuracy: "+accuracy);
    results.dispose();*/
}
function createModel(inputs, multilayer = false) {
    // Create a sequential model
    const model = tf.sequential();
    if(multilayer){
        model.add(tf.layers.dense({inputShape: [inputs.shape[1]], activation: 'sigmoid', units: 4, useBias: true}));
        model.add(tf.layers.dense({units: 1, activation: 'sigmoid', useBias: true}));
    }else{
        model.add(tf.layers.dense({inputShape: [inputs.shape[1]], activation: 'sigmoid', units: 1, useBias: true}));
    }
    return model;
}
async function trainModel(model, inputs, labels, epochs=1) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    const batchSize = 32;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
    });
}

(async function () {
    const parser = new argparse.ArgumentParser();
    parser.addArgument('--task', {
        action: 'store',
        default: 'SIMMATRIX',
        choices: ['IMAGE', 'TRAIN_NN','TRAINPERCEPTRON', 'EVALMODEL', 'SIMMATRIX'],
        help: 'Task to run, IMAGE to generate image from labeled data, TRAINPERCEPTRON to run perceptron training, EVALMODEL to load trained model from file and run, SIMMATRIX to write similarity matrix to file.'
    });
    parser.addArgument('--metric', {
        action: 'store',
        default: 'DICEsquared',
        choices: ['DICE', 'DICEsquared', 'COSINE', 'KUMAR'],
        help: 'Similarity metric to use.'
    });
    parser.addArgument('--append', {
        action: 'store',
        default: 'ING',
        help: 'String to append to filenames when searching/generating files.'
    });
    parser.addArgument('--epochs', {
        type: 'int',
        defaultValue: 1,
        help: 'Number of epochs to train the model for'
    });
    /*parser.addArgument('--savePath', {
        type: 'string',
        defaultValue: './models',
        help: 'Directory to which the decoder part of the VAE model will ' +
            'be saved after training. If the directory does not exist, it will be ' +
            'created.'
    });*/

    const args = parser.parseArgs();

    console.info('Training using GPU.');
    tf = require('@tensorflow/tfjs-node-gpu');
    const di = { tf: tf};
    sm = sm(di);
    fe = fe(di);
    ca = ca(di);

    const csvConfig = {
        Date: {
            dtype: 'string',
            fextractors: [
                fe.date,
            ],
        },
        Name: {
            dtype: 'string',
            fextractors: [
                fe.bigrams,
            ],
        },
        Description: {
            dtype: 'string',
            fextractors: [
                fe.bigrams,
            ],
        },
        Amount: {
            dtype: 'float32',
            fextractors: [
                fe.amount
            ],
        }
    }
    const csvUrl = __dirname + '/data/example3_puntcomma_delimited.csv';
    if(args.task=="IMAGE"){
        await runGenerateImageResult(csvUrl, csvConfig, args);
    }else if(args.task=="TRAINPERCEPTRON" || args.task=="TRAIN_NN"){
        await runPerceptron(csvUrl, csvConfig, args);
    }else if(args.task=="EVALMODEL"){
        await evalPerceptron(csvUrl, csvConfig, args);
    }else if(args.task=="SIMMATRIX"){
        await runSimmatrix(csvUrl, csvConfig, args);
    }
})();


