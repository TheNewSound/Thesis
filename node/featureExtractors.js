const moment = require('moment');

function buildBigrams(str) {
    for (let i = 0; i < str.length - 1; i++) {
        const bigram = str.substr(i, 2);
        if (this.bigramContainer.indexOf(bigram) === -1) {
            this.bigramContainer.push(bigram);
        }
    }
}

function str2bigram(str) {
    const array = new Array(this.bigramContainer.length).fill(0);
    for (let i = 0; i < str.length - 1; i++) {
        const bigramIndex = this.bigramContainer.indexOf(str.substr(i, 2));
        array[bigramIndex]++;
    }
    return array;
}

module.exports = (di) => {
    const obj = {};
    obj.bigrams = (input) => {
        const tf = di.tf;
        const reducer = (a, c) => a + c;
        let r = input.map(s => typeof s !== "undefined" ? s.toString().replace(/\s+/g, '') : ""); //.reduce(reducer)
        let params = {bigramContainer: []};
        r.forEach(buildBigrams.bind(params));
        if(tf === 'undefined' || tf == null){
            console.log(this);
            return r.map(str2bigram.bind(params));
        }
        return tf.tensor2d(r.map(str2bigram.bind(params)), [input.length, params.bigramContainer.length]); //, 'int32');
    };
    obj.date = (input) => {
        let r = [[], [], [], [], [], []];

        input.forEach(i => {
            const m = moment(i.toString(), ["YYYYMMDD","DD-MM-YYYY","DD.MM.YYYY"], true);
            r[0].push(m.year());
            r[1].push(m.month());
            r[2].push(m.date());
            r[3].push(m.week());
            r[4].push(m.day());
            r[5].push(m.quarter());
        });
        return r;
    };
    obj.default = (input) => {
        return input;
    };
    obj.amount = (input) => {
        return input.map(i => Math.abs(parseFloat(i.toString().replace(/,/,'.'))));
    };
    return obj;
};
