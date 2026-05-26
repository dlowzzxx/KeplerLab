const http = require("http");
const fs = require("fs");
const path = require("path");

const root = path.resolve(process.env.PINN_ROOT || __dirname);
const port = Number(process.env.PORT || 8000);

const types = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
};

function send(res, status, body, type = "text/plain; charset=utf-8") {
    res.writeHead(status, {
        "Content-Type": type,
        "Cache-Control": "no-store",
    });
    res.end(body);
}

function redirect(res, location) {
    res.writeHead(302, {
        "Location": location,
        "Cache-Control": "no-store",
    });
    res.end();
}

const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://localhost:${port}`);
    let pathname = decodeURIComponent(url.pathname);

    if (pathname === "/") {
        redirect(res, "/web/");
        return;
    }
    if (pathname.endsWith("/")) pathname += "index.html";

    const filePath = path.normalize(path.join(root, pathname));
    if (!filePath.startsWith(root)) {
        send(res, 403, "Forbidden");
        return;
    }

    fs.readFile(filePath, (err, data) => {
        if (err) {
            send(res, 404, `Not found: ${pathname}`);
            return;
        }

        send(res, 200, data, types[path.extname(filePath)] || "application/octet-stream");
    });
});

server.listen(port, () => {
    console.log(`KeplerLab: http://localhost:${port}/web/`);
});
