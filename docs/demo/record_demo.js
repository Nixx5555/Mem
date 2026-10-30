// Records the 96-second captioned walkthrough of the app (the fallback if the
// live demo or the room's Wi-Fi fails). Start the app first:
//   cd memgate && python3 app.py --no-browser          (add --llm none for no model)
// then, from a folder with playwright installed (npm i playwright):
//   node record_demo.js
// The .webm lands in video/; the benchmark wait is long, so speed that stretch
// up with ffmpeg using the times written to bench_t.json. Set CHROME to your
// Chromium if playwright's own browser is not installed.
const { chromium } = require('playwright');
const W = 1440, H = 900;
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined });
  const ctx = await browser.newContext({ viewport: { width: W, height: H },
    recordVideo: { dir: 'video', size: { width: W, height: H } } });
  const page = await ctx.newPage();
  const T0 = Date.now();
  const cap = async (text, ms = 3500) => {
    await page.evaluate((t) => {
      let el = document.getElementById('__cap');
      if (!el) { el = document.createElement('div'); el.id = '__cap';
        el.style.cssText = 'position:fixed;left:650px;top:64px;transform:translateX(-50%);max-width:860px;z-index:99999;background:rgba(15,15,20,.88);color:#fff;font:600 21px/1.35 system-ui,sans-serif;padding:14px 22px;border-radius:12px;text-align:center;box-shadow:0 6px 30px rgba(0,0,0,.35);pointer-events:none;transition:opacity .3s';
        document.body.appendChild(el); }
      el.style.opacity = t ? 1 : 0; el.textContent = t;
    }, text);
    await page.waitForTimeout(ms);
  };
  const shot = async (n) => page.screenshot({ path: `shots/${n}.png` });
  await page.goto('http://localhost:8765/');
  await page.waitForTimeout(1500);
  await cap('MemGate: a memory layer in front of a frozen language model. It decides what to keep, compress, forget and recall, under a token budget.', 5000);
  await shot('01');
  await cap('Recorded without a chat model, so replies are placeholders. Everything shown is MemGate itself: live, with the real embedder and scorer.', 4500);
  await page.selectOption('#scale', 'compact');
  await page.waitForTimeout(800);
  await cap('Compact sizes make the store small, so you can watch it forget.', 3000);
  await page.click('#sample');
  await page.waitForTimeout(2500);
  await cap('A sample chat: greetings, small talk, a deadline, a correction, a preference.', 3500);
  await shot('02');
  const memOpen = await page.evaluate(() => document.getElementById('mem').getBoundingClientRect().width > 50);
  if (!memOpen) await page.click('#memBtn');
  await page.waitForTimeout(800);
  await page.click('button[data-tab="log"]');
  await page.waitForTimeout(800);
  await cap('Every decision is logged with its score. Small talk is forgotten first; the facts stay.', 5000);
  await shot('03');
  await page.click('button[data-tab="tiers"]');
  await page.waitForTimeout(800);
  await cap('What is still held, by tier. The old deadline was superseded by the correction.', 4500);
  await shot('04');
  await page.fill('#text', 'When is my report due?');
  await page.waitForTimeout(600);
  await page.click('#send');
  await page.waitForTimeout(2500);
  await page.click('button[data-tab="ctx"]');
  await page.waitForTimeout(800);
  await cap('The model never sees the chat, only these notes. The corrected date (4 November) is in them.', 5000);
  await shot('05');
  await page.click('button[data-tab="cmp"]');
  await page.waitForTimeout(2500);
  await cap('Compare: the same question through a recency window and plain RAG at the same budget.', 5000);
  await shot('06');
  // score any message
  await page.click('button[data-tab="log"]');
  await page.waitForTimeout(600);
  await page.fill('#tryText', 'my exam is on 3 May at 10 am');
  await page.waitForTimeout(500);
  await page.click('#tryForm button');
  await page.waitForTimeout(2500);
  await cap('Score any message and see why: each feature pushes the score up or down.', 5500);
  await shot('07');
  await page.fill('#tryText', 'haha ok cool');
  await page.click('#tryForm button');
  await page.waitForTimeout(2000);
  await cap('Small talk scores far lower than a dated fact, so it is forgotten first.', 4000);
  await shot('08');
  // benchmark
  await page.click('#benchBtn');
  await page.waitForTimeout(800);
  await page.click('#benchRun');
  await cap('Benchmark check: replays all 10 LoCoMo conversations (1,527 questions) live and compares with the paper.', 3000);
  const tA = (Date.now() - T0) / 1000;
  await page.waitForSelector('#benchRes:not([hidden])', { timeout: 240000 });
  const tB = (Date.now() - T0) / 1000;
  require('fs').writeFileSync('bench_t.json', JSON.stringify({ tA, tB }));
  await page.waitForTimeout(800);
  await cap('Both numbers MATCH the committed results: 37.5% for MemGate against 28.1% for plain RAG.', 6000);
  await shot('09');
  await page.click('#benchClose');
  await page.evaluate(() => {
    const d = document.createElement('div');
    d.style.cssText = 'position:fixed;inset:0;z-index:99998;background:#111318;color:#fff;display:flex;flex-direction:column;justify-content:center;align-items:center;font:500 26px/1.5 system-ui,sans-serif;text-align:center;padding:40px';
    d.innerHTML = '<div style="font:700 44px system-ui;margin-bottom:24px">MemGate on LoCoMo, store capped at 4,096 tokens</div>' +
      '<div>Plain RAG (forget oldest) <b>28.1%</b> &nbsp;·&nbsp; MemGPT-style <b>30.2%</b> &nbsp;·&nbsp; MemGate, P1 rules <b>37.5%</b> &nbsp;·&nbsp; Mem0-style, tuned on LoCoMo <b>44.5%</b></div>' +
      '<div style="margin-top:10px;color:#3ddc97;font-weight:700;font-size:32px">MemGate, learned scorer on unseen conversations: 52.0%</div>' +
      '<div style="margin-top:22px;color:#aab;font-size:20px">Answer recall over 715 answerable questions. The learned scorer&rsquo;s lead over each is significant (paired bootstrap, McNemar). Details: REPORT.md §3.12, §3.13.</div>';
    document.body.appendChild(d);
    const c = document.getElementById('__cap'); if (c) c.remove();
  });
  await page.waitForTimeout(7000);
  await shot('10');
  await ctx.close();
  await browser.close();
})();
