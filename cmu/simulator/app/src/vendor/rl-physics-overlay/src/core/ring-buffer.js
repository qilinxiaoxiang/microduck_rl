export class RingBuffer {
  constructor(capacity = 240) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError("capacity must be a positive integer");
    }
    this.capacity = capacity;
    this.items = new Array(capacity);
    this.start = 0;
    this.length = 0;
  }

  push(value) {
    const index = (this.start + this.length) % this.capacity;
    this.items[index] = value;
    if (this.length === this.capacity) {
      this.start = (this.start + 1) % this.capacity;
    } else {
      this.length += 1;
    }
  }

  latest() {
    if (!this.length) return undefined;
    return this.items[(this.start + this.length - 1) % this.capacity];
  }

  toArray() {
    return Array.from({ length: this.length }, (_, offset) => (
      this.items[(this.start + offset) % this.capacity]
    ));
  }

  clear() {
    this.start = 0;
    this.length = 0;
  }
}
