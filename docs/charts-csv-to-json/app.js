import fs from 'fs';
import Papa from 'papaparse'

const filepath = 'input/data.csv'
const resultFilepath = '../javascripts/benchmark.json'
const header = 'engine,batch,sequence,speed'

function writeResult(result) {
    try {
        fs.writeFileSync(resultFilepath, result);
        console.log('csv conversion has been saved!');
    } catch (error) {
        console.error(`an error occurred while writing the json file :\n
            ${error}`);
    }
}

function convertCSVtoJSON (){
    try {
        const csv = fs.readFileSync(filepath, 'utf8');
        const csvWithHeader = `${header}\n${csv.trim()}`;

        Papa.parse(csvWithHeader, {
            complete: function (json) {
                console.log('csv conversion complete!');
                writeResult(JSON.stringify(json.data))
            },
            error: function (error) {
                console.error(`an error occurred while parsing the csv file :\n
                ${JSON.stringify(error)}`);
            },
            delimiter: ',',
            header: true,
            dynamicTyping: true,
        })
    } catch (error) {
        console.error(`an error occurred while reading the csv file :\n
        ${JSON.stringify(error)}`);
    }
}

convertCSVtoJSON();