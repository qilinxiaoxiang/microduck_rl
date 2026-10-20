import AppKit
import Foundation

struct Configuration: Decodable {
    let appDirectory: String
    let nodeExecutable: String
    let port: Int
}

final class Launcher: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var config: Configuration!
    var service: Process?
    var output: FileHandle?
    var starting = false
    var pendingPage: String?
    var buttons: [NSButton] = []
    var shutdownButton: NSButton!
    var shutdownTask: Process?
    let powerStatus = NSTextField(wrappingLabelWithString: "关机后等它不再响应，再断电。")
    // Live A7Z status: one SSH session runs duck-telemetry.py there, which prints a JSON line every 0.5 s.
    let robotHost = ProcessInfo.processInfo.environment["MICRODUCK_ROBOT_HOST"] ?? "a7z"  // override for testing on another host
    var telemetry: Process?
    var telemetryBuffer = Data()
    var telemetryPaused = false   // the shutdown script is running
    var expectingOff = false      // the shutdown button finished, so silence is expected
    var online = false
    var lastLine: Date?, lostSince: Date?
    var lastRecord: [String: Any]?
    var lastBoot: String?, resetNote = ""
    let linkStatus = NSTextField(labelWithString: "○ 正在连接 A7Z…")
    let healthText = NSTextField(wrappingLabelWithString: "")
    let status = NSTextField(wrappingLabelWithString: "点击一个场景，自动启动并在 Chrome 中打开。")
    let quality = NSSegmentedControl(labels: ["流畅（默认）", "清晰"], trackingMode: .selectOne, target: nil, action: nil)
    var base: URL { URL(string: "http://127.0.0.1:\(config.port)")! }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        let menu = NSMenu(), applicationMenu = NSMenu(), item = NSMenuItem()
        applicationMenu.addItem(withTitle: "退出 Microduck", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.submenu = applicationMenu; menu.addItem(item); NSApp.mainMenu = menu
        do {
            config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: Bundle.main.url(forResource: "configuration", withExtension: "json")!))
        } catch {
            let alert = NSAlert(); alert.messageText = "启动器配置丢失"; alert.informativeText = "请重新运行 desktop/install.py 安装启动器。"; alert.runModal(); NSApp.terminate(nil); return
        }
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 800), styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "Microduck"; window.isReleasedWhenClosed = false
        let content = window.contentView!, stack = NSStackView()
        stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 15
        stack.translatesAutoresizingMaskIntoConstraints = false; content.addSubview(stack)
        NSLayoutConstraint.activate([stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 32), stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -32), stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 28)])
        let heading = NSStackView(); heading.orientation = .horizontal; heading.spacing = 12
        let icon = NSImageView(image: NSImage(contentsOf: Bundle.main.url(forResource: "Microduck", withExtension: "icns")!)!)
        icon.widthAnchor.constraint(equalToConstant: 50).isActive = true; icon.heightAnchor.constraint(equalToConstant: 50).isActive = true
        heading.addArrangedSubview(icon)
        let title = NSTextField(labelWithString: "Microduck")
        title.font = .systemFont(ofSize: 30, weight: .bold); heading.addArrangedSubview(title); stack.addArrangedSubview(heading)
        let subtitle = NSTextField(labelWithString: "选一个场景，开始玩。")
        subtitle.font = .systemFont(ofSize: 16); subtitle.textColor = .secondaryLabelColor; stack.addArrangedSubview(subtitle)
        addScene("普通仿真", detail: "自由走动、踢球、坐下和站起。", tag: 0, to: stack)
        addScene("叼小球", detail: "自动完成低头、叼起、保持和放下。", tag: 1, to: stack)
        addScene("叼笔", detail: "按 G 自动叼取，按 E 放下；W/A/S/D 移动。", tag: 2, to: stack)
        quality.selectedSegment = 0; quality.toolTip = "只调整画面精细度。两档使用相同的物理步长和碰撞检查。"
        stack.addArrangedSubview(quality)
        let tip = NSTextField(wrappingLabelWithString: "切到其他标签页时，仿真自动暂停；回来继续。\n运行时让仿真页面保持可见。")
        tip.font = .systemFont(ofSize: 13); tip.textColor = .secondaryLabelColor; stack.addArrangedSubview(tip)
        status.font = .systemFont(ofSize: 13); status.textColor = .secondaryLabelColor; stack.addArrangedSubview(status)
        let robot = NSTextField(labelWithString: "真机")
        robot.font = .systemFont(ofSize: 16, weight: .semibold); stack.addArrangedSubview(robot)
        linkStatus.font = .systemFont(ofSize: 14, weight: .semibold); linkStatus.textColor = .secondaryLabelColor; stack.addArrangedSubview(linkStatus)
        healthText.font = .systemFont(ofSize: 12); stack.addArrangedSubview(healthText)
        addScene("真机监视", detail: "温度、电压曲线和鸭子的实时姿态，在 Chrome 打开。", tag: 3, to: stack)
        shutdownButton = NSButton(title: "优雅关机 · A7Z 大脑", target: self, action: #selector(shutdownRobot(_:)))
        shutdownButton.bezelStyle = .rounded; shutdownButton.controlSize = .large; shutdownButton.font = .systemFont(ofSize: 16, weight: .medium)
        shutdownButton.heightAnchor.constraint(equalToConstant: 36).isActive = true; stack.addArrangedSubview(shutdownButton)
        powerStatus.font = .systemFont(ofSize: 13); powerStatus.textColor = .secondaryLabelColor; stack.addArrangedSubview(powerStatus)
        let footer = NSTextField(wrappingLabelWithString: "不用打开终端。关闭此窗口可继续玩；⌘Q 退出会停止本应用启动的服务。首次加载需要联网。")
        footer.font = .systemFont(ofSize: 11); footer.textColor = .tertiaryLabelColor; stack.addArrangedSubview(footer)
        for view in stack.arrangedSubviews { view.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true }
        window.center(); window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true)
        startTelemetry()
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in self.watchTelemetry() }
        #if SNAPSHOT  // test builds only (swiftc -D SNAPSHOT): render the window to $MICRODUCK_SNAPSHOT_DIR every 2 s
        if let dir = ProcessInfo.processInfo.environment["MICRODUCK_SNAPSHOT_DIR"] {
            var n = 0
            Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
                let view = self.window.contentView!, rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)!
                view.cacheDisplay(in: view.bounds, to: rep); n += 1
                try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: "\(dir)/shot-\(n).png"))
            }
        }
        #endif
    }
    func addScene(_ name: String, detail: String, tag: Int, to stack: NSStackView) {
        let row = NSStackView(); row.orientation = .vertical; row.alignment = .leading; row.spacing = 4
        let button = NSButton(title: "打开 · \(name)", target: self, action: #selector(openScene(_:)))
        button.tag = tag; button.bezelStyle = .rounded; button.controlSize = .large; button.font = .systemFont(ofSize: 16, weight: .medium)
        button.heightAnchor.constraint(equalToConstant: 36).isActive = true
        row.addArrangedSubview(button); buttons.append(button)
        let label = NSTextField(labelWithString: detail); label.font = .systemFont(ofSize: 12); label.textColor = .secondaryLabelColor
        row.addArrangedSubview(label); stack.addArrangedSubview(row)
    }
    @objc func openScene(_ sender: NSButton) {
        guard !starting else { return }
        let pages = ["/?boot=1", "/ground-pick.html?controller=learned", "/household-pick.html?object=pen", "/robot-live.html?live=1"]
        pendingPage = pages[sender.tag] + "&quality=" + (quality.selectedSegment == 0 ? "fast" : "sharp")
        starting = true; buttons.forEach { $0.isEnabled = false }; status.stringValue = "正在准备仿真…"
        probe { ready in
            if ready { self.finishOpening() } else { self.startService() }
        }
    }
    func probe(_ completion: @escaping (Bool) -> Void) {
        var request = URLRequest(url: base.appendingPathComponent("microduck-launcher.json")); request.timeoutInterval = 1
        request.cachePolicy = .reloadIgnoringLocalCacheData
        URLSession.shared.dataTask(with: request) { data, response, _ in
            let valid = (response as? HTTPURLResponse)?.statusCode == 200 && data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: String] }?["application"] == "microduck-local-launcher-v1"
            DispatchQueue.main.async { completion(valid) }
        }.resume()
    }
    func startService() {
        let vite = URL(fileURLWithPath: config.appDirectory).appendingPathComponent("node_modules/vite/bin/vite.js")
        guard FileManager.default.fileExists(atPath: vite.path), FileManager.default.isExecutableFile(atPath: config.nodeExecutable) else {
            fail("找不到仿真依赖或 Node.js。请在仿真 app 文件夹安装依赖后，重新运行 desktop/install.py。"); return
        }
        if service?.isRunning == true { awaitReady(remaining: 80); return }
        do {
            let logDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Microduck")
            try FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
            let log = logDir.appendingPathComponent("server.log")
            if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
            output = try FileHandle(forWritingTo: log); try output?.seekToEnd()
            let process = Process(); process.executableURL = URL(fileURLWithPath: config.nodeExecutable)
            process.currentDirectoryURL = URL(fileURLWithPath: config.appDirectory)
            process.arguments = [vite.path, "--host", "127.0.0.1", "--port", String(config.port), "--strictPort"]
            process.standardOutput = output; process.standardError = output
            process.terminationHandler = { _ in DispatchQueue.main.async { if !self.starting { self.status.stringValue = "本机服务已停止。点击场景即可重新启动。" } } }
            service = process; try process.run(); awaitReady(remaining: 80)
        } catch { fail("本机服务未能启动：\(error.localizedDescription)") }
    }
    func awaitReady(remaining: Int) {
        probe { ready in
            if ready { self.finishOpening() }
            else if remaining == 0 || self.service?.isRunning != true {
                self.fail("服务没有启动成功。端口 \(self.config.port) 可能被占用；详情在 ~/Library/Logs/Microduck/server.log。")
            } else { DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.awaitReady(remaining: remaining - 1) } }
        }
    }
    func finishOpening() {
        guard let page = pendingPage, let url = URL(string: base.absoluteString + page), let chrome = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.google.Chrome") else {
            fail("找不到 Google Chrome，请先安装 Chrome。"); return
        }
        NSWorkspace.shared.open([url], withApplicationAt: chrome, configuration: NSWorkspace.OpenConfiguration()) { _, error in
            DispatchQueue.main.async {
                if let error { self.fail("Chrome 未能打开：\(error.localizedDescription)") }
                else { self.starting = false; self.buttons.forEach { $0.isEnabled = true }; self.status.stringValue = "已在 Chrome 打开。可随时回来切换场景。" }
            }
        }
    }
    @objc func shutdownRobot(_ sender: NSButton) {
        guard shutdownTask == nil, let script = Bundle.main.url(forResource: "duck-shutdown", withExtension: "sh") else { return }
        let alert = NSAlert(); alert.messageText = "关闭 MicroDuck 的 A7Z？"
        alert.informativeText = "A7Z 会执行系统关机，舵机电源不受影响。关机后要重新上电才能再开机。"
        alert.addButton(withTitle: "关机"); alert.addButton(withTitle: "取消")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/sh"); process.arguments = [script.path, robotHost]
        process.standardOutput = pipe; process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            guard let text = String(data: handle.availableData, encoding: .utf8), let line = text.split(separator: "\n").last else { return }
            DispatchQueue.main.async { self.powerStatus.stringValue = String(line) }
        }
        process.terminationHandler = { finished in
            DispatchQueue.main.async {
                pipe.fileHandleForReading.readabilityHandler = nil
                if let rest = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8), let line = rest.split(separator: "\n").last { self.powerStatus.stringValue = String(line) }
                // Exit 2: the A7Z was already unreachable before any command was sent.
                self.powerStatus.textColor = [0: NSColor.systemGreen, 2: NSColor.systemOrange][finished.terminationStatus] ?? .systemRed
                self.shutdownTask = nil; self.shutdownButton.isEnabled = true
                self.expectingOff = [0, 2].contains(finished.terminationStatus); self.telemetryPaused = false; self.startTelemetry()
            }
        }
        shutdownButton.isEnabled = false; powerStatus.textColor = .secondaryLabelColor; powerStatus.stringValue = "正在连接 A7Z…"
        // The shutdown script stops every Python program on the A7Z; pause the status stream so it does not restart one.
        telemetryPaused = true; telemetry?.terminate()
        do { shutdownTask = process; try process.run() }
        catch { shutdownTask = nil; shutdownButton.isEnabled = true; telemetryPaused = false; startTelemetry(); powerStatus.textColor = .systemRed; powerStatus.stringValue = "关机脚本没能启动：\(error.localizedDescription)" }
    }
    func startTelemetry() {
        guard telemetry == nil, !telemetryPaused, let script = Bundle.main.url(forResource: "duck-telemetry", withExtension: "py"), let body = try? Data(contentsOf: script) else { return }
        let process = Process(), output = Pipe(), input = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        // ServerAlive closes a session whose A7Z vanished without a goodbye (reset, power cut) within about 4 s.
        process.arguments = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
                             robotHost, "exec nice -n 10 python3 -u - duck-telemetry"]
        process.standardInput = input; process.standardOutput = output; process.standardError = FileHandle.nullDevice
        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            DispatchQueue.main.async { if self.telemetry === process { self.receive(data) } }
        }
        process.terminationHandler = { _ in
            DispatchQueue.main.async {
                output.fileHandleForReading.readabilityHandler = nil
                if self.telemetry === process { self.telemetry = nil; self.telemetryBuffer.removeAll() }
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.startTelemetry() }
            }
        }
        do {
            try process.run(); telemetry = process
            input.fileHandleForWriting.write(body); try? input.fileHandleForWriting.close()
        } catch { DispatchQueue.main.asyncAfter(deadline: .now() + 5) { self.startTelemetry() } }
    }
    func receive(_ data: Data) {
        telemetryBuffer.append(data)
        while let end = telemetryBuffer.firstIndex(of: 10) {
            let line = telemetryBuffer[telemetryBuffer.startIndex..<end]
            telemetryBuffer.removeSubrange(telemetryBuffer.startIndex...end)
            guard let record = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else { continue }
            if let boot = record["boot"] as? String {
                if let previous = lastBoot, previous != boot {
                    resetNote = "A7Z 在 \(clock(lostSince ?? Date())) 前后重启过（boot 变了）。"
                    logEvent("reboot", record)
                }
                lastBoot = boot
            }
            if !online { online = true; expectingOff = false; lostSince = nil; NSApp.dockTile.badgeLabel = nil; logEvent("online", record) }
            lastLine = Date(); lastRecord = record
            render(record, stale: false)
        }
    }
    func watchTelemetry() {
        let silent = lastLine.map { Date().timeIntervalSince($0) } ?? .infinity
        if online && silent > 1.5 {
            online = false; lostSince = lastLine
            logEvent(expectingOff || telemetryPaused ? "off" : "lost", lastRecord)
            if !expectingOff && !telemetryPaused { NSApp.dockTile.badgeLabel = "!"; NSApp.requestUserAttention(.criticalRequest) }
            telemetry?.terminate()  // reconnect from scratch rather than wait on a dead TCP session
            if let record = lastRecord { render(record, stale: true) }
        }
        if online {
            let up = Int((lastRecord?["up"] as? Double) ?? 0)
            setLink("● A7Z 在线 · 开机 \(up / 3600 > 0 ? "\(up / 3600) 小时 " : "")\(up / 60 % 60) 分 · 每 0.5 秒刷新", .systemGreen)
        } else if telemetryPaused {
            setLink("○ 关机中，状态监视暂停", .secondaryLabelColor)
        } else if expectingOff {
            setLink("○ A7Z 已关机（按钮关机）。重新上电后自动连上。", .secondaryLabelColor)
        } else if let since = lostSince {
            setLink("● A7Z 失联 \(Int(Date().timeIntervalSince(since))) 秒（\(clock(since)) 起）· 正在重连", .systemRed)
        } else {
            setLink("○ 正在连接 A7Z…（还没连上过）", .secondaryLabelColor)
        }
    }
    func setLink(_ text: String, _ color: NSColor) { linkStatus.stringValue = text; linkStatus.textColor = color }
    func render(_ r: [String: Any], stale: Bool) {
        // Two-line summary; the charts and the pose live on /robot-live.html.
        var parts: [String] = []
        let temps = ((r["temps"] as? [String: Double]) ?? [:]).filter { !$0.key.contains("idle") }
        if let hottest = temps.values.max() { parts.append(String(format: "SoC 最高 %.1f °C", hottest) + (hottest >= 80 ? "（偏热）" : "")) }
        if let mhz = r["mhz"] as? Int {
            var cpu = "CPU \(mhz) MHz"
            if let cap = r["mhz_cap"] as? Int, let hw = r["mhz_hw"] as? Int, cap < hw { cpu += "（降频）" }
            parts.append(cpu)
        }
        let brain = (r["brain"] as? [String: Any]) ?? [:]
        if let lo = brain["servo_v_min"] as? Double, let hi = brain["servo_v_max"] as? Double { parts.append(String(format: "舵机 %.1f–%.1f V", lo, hi)) }
        parts.append(brain["running"] as? Bool == true ? "大脑 \((brain["mode"] as? String) ?? "?")" : "大脑没在运行")
        var text = parts.joined(separator: " · ")
        if !resetNote.isEmpty { text += "\n" + resetNote }
        healthText.stringValue = stale ? "最后读数（\(clock(Date(timeIntervalSince1970: (r["t"] as? Double) ?? 0)))）：" + text : text
        healthText.textColor = stale ? .secondaryLabelColor : .labelColor
    }
    func clock(_ date: Date) -> String { let f = DateFormatter(); f.dateFormat = "HH:mm:ss"; return f.string(from: date) }
    func logEvent(_ event: String, _ record: [String: Any]?) {
        // ~/Library/Logs/Microduck/a7z.log keeps connect/lose/reboot moments with the last reading, for reset diagnosis.
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Microduck/a7z.log")
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let body = record.flatMap { try? JSONSerialization.data(withJSONObject: $0) }.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        let line = Data("\(ISO8601DateFormatter().string(from: Date())) \(event) \(body)\n".utf8)
        if let handle = try? FileHandle(forWritingTo: url) { handle.seekToEndOfFile(); handle.write(line); try? handle.close() }
        else { try? line.write(to: url) }
    }
    func fail(_ message: String) { starting = false; buttons.forEach { $0.isEnabled = true }; status.stringValue = message }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { window.makeKeyAndOrderFront(nil); return true }
    func applicationWillTerminate(_ notification: Notification) { telemetryPaused = true; telemetry?.terminate(); shutdownTask?.terminate(); if service?.isRunning == true { service?.terminate() }; try? output?.close() }
}
let app = NSApplication.shared
let delegate = Launcher(); app.delegate = delegate
app.run()
