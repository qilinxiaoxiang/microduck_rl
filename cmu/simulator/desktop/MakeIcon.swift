import AppKit
let directory = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
// Package the square artwork at macOS icon sizes inside the standard rounded-square tile.
guard let drawing = NSImage(contentsOfFile: CommandLine.arguments[2]) else { fatalError("Missing icon artwork") }
for logical in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = logical * scale
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        let tile = NSRect(x: 0, y: 0, width: pixels, height: pixels).insetBy(dx: CGFloat(pixels) * 0.098, dy: CGFloat(pixels) * 0.098)
        NSBezierPath(roundedRect: tile, xRadius: tile.width * 0.225, yRadius: tile.width * 0.225).addClip()
        drawing.draw(in: tile)
        NSGraphicsContext.restoreGraphicsState()
        let suffix = scale == 2 ? "@2x" : ""
        try bitmap.representation(using: .png, properties: [:])!.write(to: directory.appendingPathComponent("icon_\(logical)x\(logical)\(suffix).png"))
    }
}
