const express = require('express');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());
app.use(express.static('app'));

app.post('/start', (req, res) => {
    const prompt = req.body.prompt;
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    const process = spawn('python', ['test.py', prompt]);

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
        stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    process.on('close', (code) => {
        if (stderr) {
            console.error('Stderr:', stderr);
        }

        try {
            const output = JSON.parse(stdout);
            if (typeof output.story === 'string') {
                const innerMatch = output.story.match(/\{[\s\S]*\}/); // capture inner JSON object
                if (innerMatch) {
                    const inner = JSON.parse(innerMatch[0]);
                    res.json({
                        story: inner.story,
                        options: inner.options
                    });
                    return;
                }
            }

            res.json(output);
        } catch (e) {
            console.error(' Failed to parse JSON:', e.message);
            res.status(200).json({ raw: stdout });
        }
    });
});

app.post("/generate-image", (req, res) => {
    const story = req.body.story;
    if (!story) return res.status(400).json({ error: "Story text is required" });

    const process = spawn("python", ["imagegen.py", story]);

    let stdout = "", stderr = "";
    process.stdout.on("data", (data) => stdout += data.toString());
    process.stderr.on("data", (data) => stderr += data.toString());

    process.on("close", () => {
        if (stderr) console.error("imagegen.py stderr:", stderr);
        try {
            const output = JSON.parse(stdout);
            res.json({ imageUrl: output.imageUrl });
        } catch (e) {
            console.error(" Failed to parse imagegen output:", e.message);
            res.status(500).json({ error: "Image generation failed", raw: stdout });
        }
    });
});


app.listen(5000, () => {
    console.log('listening in 5000')
});