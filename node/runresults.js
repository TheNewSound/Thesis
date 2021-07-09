const util = require('util')
const childProcess = require('child_process')
const exec = util.promisify(childProcess.exec)

const metrics = ["KUMAR"]; //,"KUMAR"
const ING = "ING";
let outputs = []
const loop = async _ => {
    for await (metric of metrics) {
        console.log("METRIC: " + metric);
        //const dir = await exec('ls -al');
        //console.log("DIR: " + dir.stdout);
        outputs.push("METRIC: " + metric);
        try {
            let results = exec('/snap/bin/node index.js --task=SIMMATRIX --metric=' + metric + ' --append=' + ING + '_' + metric + '');
        console.log(results.stdout);
        //results = await exec('/snap/bin/node index.js --task=TRAINPERCEPTRON --metric=' + metric + ' --append=' + ING + '_' + metric + '');
        //console.log(results.stdout);
        results = await exec('/snap/bin/node index.js --task=EVALMODEL --metric=' + metric + ' --append=' + ING + '_' + metric + '');
        console.log(results.stdout);
        results = await exec('python3 ../python/main.py --method=PERCEPTRON --out-name=' + ING + '_' + metric + ' --validation-file=../node/data/example2_puntcomma_delimited.csv --similarity-file=../node/output/SimilarityMatrix_' + ING + '_' + metric + '.bin --latex=True');
        console.log(results.stdout);
        outputs.push(results.stdout);
        results = await exec('python3 ../python/main.py --method=HACsingle --out-name=' + ING + '_' + metric + ' --validation-file=../node/data/example2_puntcomma_delimited.csv --similarity-file=../node/output/SimilarityMatrix_' + ING + '_' + metric + '.bin --latex=True');
        console.log(results.stdout);
        outputs.push(results.stdout);
        results = await exec('python3 ../python/main.py --method=HACcomplete --out-name=' + ING + '_' + metric + ' --validation-file=../node/data/example2_puntcomma_delimited.csv --similarity-file=../node/output/SimilarityMatrix_' + ING + '_' + metric + '.bin --latex=True');
        console.log(results.stdout);
        outputs.push(results.stdout);
        results = await exec('python3 ../python/main.py --method=HDBSCAN --out-name=' + ING + '_' + metric + ' --validation-file=../node/data/example2_puntcomma_delimited.csv --similarity-file=../node/output/SimilarityMatrix_' + ING + '_' + metric + '.bin --latex=True');
        console.log(results.stdout);
        outputs.push(results.stdout);
        results = await exec('/snap/bin/node index.js --task=IMAGE --metric=' + metric + ' --append=' + ING + '_' + metric + '');
        console.log(results.stdout);
        }catch (e){
            console.log(e)
        }
    }
}

/*let result = childProcess.spawnSync('/snap/bin/node',['index.js','--task=TRAINPERCEPTRON','--metric=DICE','--append=ING_DICE'])
console.log(result)
console.log(result.stdout.toString())
console.log(result.stderr.toString())*/

loop();
for (output of outputs){
    console.log(output);
}
