function sortIndices(arr){
    return Array.from(Array(arr.length).keys()).sort((a, b) => { return arr[b]-arr[a]; })
}

module.exports = (di) => {
    const obj = {};
    obj.sortSimilarityMatrix = (input) => {
        const tf = di.tf;
        /*let indices_sorted = sortIndices(input.gather(60).arraySync());
        const k = Math.min(Math.ceil(input.shape[0]/100),input.shape[0]); // K = shape[0]/100 with maximum of shape[0].
        let isPlaced = new Array(input.shape[0]).fill(false);
        const {values, indices} = tf.topk(input,k);
        indices_sorted.forEach((item, position)=>{

        });
        values.print();
        indices.print();
        //console.log(indices_sorted);*/
        //TODO more robust (using topk, and swapping segments)
        let indices_sorted = sortIndices(input.gather(200).arraySync());
        let indices = tf.tensor1d(indices_sorted, 'int32');
        let ret = input.gather(indices,0).gather(indices,1);
        return ret;

    };
    obj.simpleCluster = async (input, threshold = 0.7, getFinalClusters = false) => {
        const tf = di.tf;
        let ret;
        let timer = process.hrtime();
        if(getFinalClusters) {
            const clusteredMatrix = tf.linalg.bandPart(input.sub(threshold).maximum(0).cast('bool'), 0, -1); // setting lower triangle to 0 to save CPU processing time below
            const coordinates = await tf.whereAsync(clusteredMatrix);
            ret = new Array(input.shape[0]).fill(0);
            let i = 1;
            coordinates.arraySync().forEach((value) => {
                if (ret[value[0]] == 0) {
                    ret[value[0]] = i++;
                } else if (ret[value[1]] == 0) {
                    ret[value[1]] = ret[value[0]];
                }
            });
            let d = process.hrtime(timer);
            console.info("Calculating final clusteredMatrix + Grabbing final clusters: %ds %dms", d[0], d[1] / 1000000);
            return { data: clusteredMatrix, numClusters: i, clusters: ret };
        }else {
            const clusteredMatrix = input.sub(threshold).maximum(0).cast('bool');
            let d = process.hrtime(timer);
            console.info("Calculating final clusteredMatrix: %ds %dms", d[0], d[1] / 1000000);
            return {data: clusteredMatrix};
        }
    };
    obj.simpleCluster2 = async (input, threshold = 0.7, getFinalClusters = false) => {
        const tf = di.tf;
        let ret;
        let timer = process.hrtime();
            if(getFinalClusters) {
                const clusteredMatrix = tf.linalg.bandPart(input.cast('bool'), 0, -1); // setting lower triangle to 0 to save CPU processing time below
                const coordinates = await tf.whereAsync(clusteredMatrix);
                    ret = new Array(input.shape[0]).fill(0);
                    let i = 1;
                    coordinates.arraySync().forEach((value) => {
                        if (ret[value[0]] == 0) {
                            ret[value[0]] = i++;
                        } else if (ret[value[1]] == 0) {
                            ret[value[1]] = ret[value[0]];
                        }
                    });
                    let d = process.hrtime(timer);
                    console.info("Calculating final clusteredMatrix + Grabbing final clusters: %ds %dms", d[0], d[1] / 1000000);
                    return { data: clusteredMatrix, numClusters: i, clusters: ret };
            }else {
                const clusteredMatrix = input.greater(threshold).cast('int32').mul(255);
                let d = process.hrtime(timer);
                console.info("Calculating final clusteredMatrix: %ds %dms", d[0], d[1] / 1000000);
                return {data: clusteredMatrix};
            }

    };
    /*obj.simpleCluster = async (input, threshold = 0.7, getFinalClusters = false) => {
        const tf = di.tf;
        let ret;
        const clusteredMatrix = input.sub(threshold).maximum(0).mul(512).cast('bool');
        if(getFinalClusters) {
            const coordinates = await tf.whereAsync(clusteredMatrix);
            ret = new Array(input.shape[0]).fill(0);
            let i = 1;
            coordinates.arraySync().forEach((value) => {
                if (ret[value[0]] == 0) {
                    ret[value[0]] = i++;
                } else if (ret[value[1]] == 0) {
                    ret[value[1]] = ret[value[0]];
                }
            });
            return { data: clusteredMatrix, numClusters: i, clusters: ret };
        }
        return { data: clusteredMatrix };
    };*/
    obj.calculateAccuracy = (matrix, validation) => {
        const tf = di.tf;
        const resultT = tf.math.confusionMatrix(validation.as1D(), matrix.as1D(),2);
        const result = resultT.arraySync();
        console.log('[tn, fp]   ['+result[0][0]+','+result[0][1]+']');
        console.log('[fn, tp]   ['+result[1][0]+','+result[1][1]+']');
        resultT.dispose();
        return (result[0][0]+result[1][1])/(result[0][0]+result[0][1]+result[1][0]+result[1][1]);
    };
    return obj;
};
