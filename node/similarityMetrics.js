module.exports = (di) => {
    const obj = {};

    // DICE similarity metric (likely only works in browser due to WebGL kernel backend).
    obj.DICE = (matrix, options={}) => {
        const tf = di.tf;
        let t = matrix;
        return tf.tidy(() => {
            if ('removeCommon' in options && options.removeCommon) {
                t = matrix.sub(matrix.min(0), 0); // remove commonalities (bigrams that every entry has)
            }

            let l;
            if (t.shape.length > 1) {
                l = tf.sum(t, 1); // sum bigrams along axis 1 to get string length
            } else {
                l = t;
            }
            const sumMatrix = l.expandDims(-1).add(l).cast('float32');  // create sum matrix

            //WebGL kernel
            /*const intersectionKernel = inputShape => {
                const variableNames = ['X'];
                const outputShape = [inputShape[0], inputShape[0]];
                const userWidth = inputShape[1];
                const userCode = `
                void main() {
                    ivec2 coords = getOutputCoords();
                    float sum = 0.0;
                    for(int i=0; i<${userWidth}; i++) {
                        sum += min(getX(coords.y, i),getX(coords.x, i));
                    }
                    setOutput(sum);
                }`;
                return {variableNames, outputShape, userCode};
            };

            const intersection = intersectionKernel(t.shape);

            const result = tf.env().compileAndRun(intersection, [t]).mul(2).div(sumMatrix);*/

            const noemer = [];
            if (t.shape.length > 1) {
                for(let i=0;i<t.shape[0];i++){
                    noemer[i] = tf.tidy(()=>{
                        const row = tf.slice2d(t, i, 1).squeeze();
                        return t.minimum(row).sum(1);
                    });
                }
            } else {
                const values = t.dataSync();
                for (let i = 0; i < t.shape[0]; i++) {
                    noemer[i] = t.minimum(values[i]);
                }
            }
            const updiv = tf.stack(noemer).cast('float32');
            updiv.print();
            const r = updiv.mul(2).div(sumMatrix);
            const nan = r.isNaN();
            const result = tf.where(nan, nan.toFloat(), r).maximum(0.0).minimum(1.0);

            nan.dispose();

            if ('removeCommon' in options && options.removeCommon) {
                t.dispose();
            }
            l.dispose();
            sumMatrix.dispose();
            return result;
        });
    };

    //Dice Squared similarity metric
    obj.VH = (matrix, options={}) => {
        const tf = di.tf;
        let t = matrix;
        return tf.tidy(() => {
            if ('removeCommon' in options && options.removeCommon) {
                t = matrix.sub(matrix.min(0), 0); // remove commonalities (minimum-values that every entry has)
            }

            let l;
            if (matrix.shape.length > 1) {
                l = t.square().sum(1); // lengthVector
            } else {
                l = t.square(); // lengthVector
            }

            const m = l.expandDims(-1).add(l); // maxMatrix

            let sim;
            if (matrix.shape.length > 1) {
                sim = tf.matMul(t, t, false, true); // simMatrix
            } else {
                sim = tf.outerProduct(t, t);
            }
            const r = sim.cast('float32').mul(2).div(m.cast('float32'));
            m.dispose();
            const nan = r.isNaN();
            const result = tf.where(nan, nan.toFloat(), r);

            nan.dispose();

            if ('removeCommon' in options && options.removeCommon) {
                t.dispose();
            }
            l.dispose();
            sim.dispose();
            r.dispose();

            return result;
        });
    };

    //Cosine similarity metric
    obj.cosine = (matrix, options={}) => {
        const tf = di.tf;
        let t = matrix;
        return tf.tidy(() => {
            if ('removeCommon' in options && options.removeCommon) {
                t = matrix.sub(matrix.min(0), 0); // remove commonalities (minimum-values that every entry has)
            }
            let l;
            if (matrix.shape.length > 1) {
                l = t.square().sum(1).sqrt(); // lengthVector
            } else {
                l = t; // lengthVector
            }
            const m = tf.outerProduct(l, l);//l.expandDims(-1).add(l); // maxMatrix

            let sim;
            if (matrix.shape.length > 1) {
                sim = tf.matMul(t, t, false, true); // simMatrix
            } else {
                sim = tf.outerProduct(t, t);
            }
            const r = sim.cast('float32').div(m.cast('float32'));
            m.dispose();
            const nan = r.isNaN();
            const result = tf.where(nan, nan.toFloat(), r).minimum(1.0).maximum(0.0);

            nan.dispose();

            if ('removeCommon' in options && options.removeCommon) {
                t.dispose();
            }
            l.dispose();
            sim.dispose();
            r.dispose();

            return result;
        });
    }

    //Kumar-Hassebrook similarity metric
    obj.kumar = (matrix, options={}) => {
        const tf = di.tf;
        let t = matrix;
        return tf.tidy(() => {
            if ('removeCommon' in options && options.removeCommon) {
                t = matrix.sub(matrix.min(0), 0); // remove commonalities (minimum-values that every entry has)
            }

            let l;
            if (matrix.shape.length > 1) {
                l = t.square().sum(1); // lengthVector
            } else {
                l = t.square(); // lengthVector
            }

            const m = l.expandDims(-1).add(l).cast('float32'); // maxMatrix

            let sim;
            if (matrix.shape.length > 1) {
                sim = tf.matMul(t, t, false, true).cast('float32'); // simMatrix
            } else {
                sim = tf.outerProduct(t, t).cast('float32');
            }
            const r = sim.div(m.sub(sim));
            m.dispose();
            const nan = r.isNaN();
            const result = tf.where(nan, nan.toFloat(), r);

            nan.dispose();

            if ('removeCommon' in options && options.removeCommon) {
                t.dispose();
            }
            l.dispose();
            sim.dispose();
            r.dispose();

            return result;
        });
    }
    return obj;
};
