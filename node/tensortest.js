let ap = require('argparse');
let parser = new ap.ArgumentParser({
    version: '0.5.0',
    addHelp: true,
    description: 'Runs tf.matMul() on a n x n matrix'
});
parser.addArgument(
    [ '-n', '--number' ],
    {
        help: 'Optional, integer',
        type: 'int',
        defaultValue: 1024,
    }
);
parser.addArgument(
    [ '-p', '--print' ],
    {
        action: 'storeTrue',
        help: 'Optional, boolean',
        defaultValue: false,
    }
);
const args = parser.parseArgs();

let tf = require('@tensorflow/tfjs-node-gpu');

const size = args.number;

console.info("Generating a %dx%d randomUniform matrix, please standby", size, size);
let timer = process.hrtime();
let matrix = tf.initializers.randomUniform({minval: 0, maxval:1, seed: 2}).apply([size,size],'float32'); // tf.initializers.zeros().apply([size,size],"float32"); //
let d = process.hrtime(timer);
console.info("Generating a %dx%d matrix took: %ds %dms", size, size, d[0], d[1] / 1000000);

if(args.print===true){
    timer = process.hrtime();
    matrix.print();
    d = process.hrtime(timer);
    console.info("Printing a %dx%d matrix took: %ds %dms", size, size, d[0], d[1] / 1000000);
}
// Initialize cuBLAS....
let temp = tf.tensor2d([0.0,1.0,2.0,3.0],[2,2],"float32");
temp.matMul(temp,false,true);


timer = process.hrtime();
let result = matrix.matMul(matrix,false,true);
d = process.hrtime(timer);
console.info("Multiplying a %dx%d matrix took: %ds %dms", size, size, d[0], d[1] / 1000000);