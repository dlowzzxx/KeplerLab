/**
 * verlet.js - Velocity Verlet integrator for the two-body problem.
 *
 * Same algorithm as the Python training code. Provides real-time
 * step-by-step integration for the animation loop, and batch
 * integration for speed benchmarking.
 *
 * Pure Keplerian two-body: a = -mur/|r|^3
 *
 * Velocity Verlet scheme:
 *   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
 *   a(t+dt) = f(x(t+dt))
 *   v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
 */

// Physical constants
const G = 6.67430e-11;       // Gravitational constant [m^3 kg^-1 s^-2]
const M_EARTH = 5.972e24;    // Earth mass [kg]
const R_EARTH = 6.371e6;     // Earth mean radius [m]
const MU = G * M_EARTH;      // Standard gravitational parameter [m^3/s^2]

class VerletIntegrator {
    constructor() {
        this.mu = MU;
    }

    /**
     * Compute pure Newtonian gravitational acceleration at position (x, y).
     * a = -mu * r / |r|^3
     *
     * @param {number} x - Position x [m]
     * @param {number} y - Position y [m]
     * @returns {{ax: number, ay: number}} Acceleration [m/s^2]
     */
    acceleration(x, y) {
        const rSq = x * x + y * y;
        const r = Math.sqrt(rSq);
        const r3 = rSq * r;

        return {
            ax: -this.mu * x / r3,
            ay: -this.mu * y / r3,
        };
    }

    /**
     * Perform a single Velocity Verlet step.
     *
     * @param {object} state - {x, y, vx, vy, ax, ay}
     * @param {number} dt - Time step [s]
     * @returns {object} New state {x, y, vx, vy, ax, ay}
     */
    step(state, dt) {
        const halfDt = 0.5 * dt;
        const halfDt2 = 0.5 * dt * dt;

        // Position update
        const xNew = state.x + state.vx * dt + state.ax * halfDt2;
        const yNew = state.y + state.vy * dt + state.ay * halfDt2;

        // New acceleration
        const acc = this.acceleration(xNew, yNew);

        // Velocity update
        const vxNew = state.vx + (state.ax + acc.ax) * halfDt;
        const vyNew = state.vy + (state.ay + acc.ay) * halfDt;

        return {
            x: xNew, y: yNew,
            vx: vxNew, vy: vyNew,
            ax: acc.ax, ay: acc.ay,
        };
    }

    /**
     * Initialize integrator state from position and velocity.
     *
     * @param {number} x0 - Initial x [m]
     * @param {number} y0 - Initial y [m]
     * @param {number} vx0 - Initial vx [m/s]
     * @param {number} vy0 - Initial vy [m/s]
     * @returns {object} Initial state with acceleration
     */
    initState(x0, y0, vx0, vy0) {
        const acc = this.acceleration(x0, y0);
        return {
            x: x0, y: y0,
            vx: vx0, vy: vy0,
            ax: acc.ax, ay: acc.ay,
        };
    }

    /**
     * Integrate multiple steps from an initial state.
     * Used for speed benchmarking.
     *
     * @param {number} x0 - Initial x [m]
     * @param {number} y0 - Initial y [m]
     * @param {number} vx0 - Initial vx [m/s]
     * @param {number} vy0 - Initial vy [m/s]
     * @param {number} dt - Time step [s]
     * @param {number} numSteps - Number of steps
     * @returns {Array<{x,y,vx,vy}>} Trajectory array
     */
    integrate(x0, y0, vx0, vy0, dt, numSteps) {
        const trajectory = [{x: x0, y: y0, vx: vx0, vy: vy0}];
        let state = this.initState(x0, y0, vx0, vy0);

        for (let i = 0; i < numSteps; i++) {
            state = this.step(state, dt);
            trajectory.push({
                x: state.x, y: state.y,
                vx: state.vx, vy: state.vy,
            });
        }
        return trajectory;
    }

    /**
     * Compute specific orbital energy.
     * epsilon = v^2/2 - mu/r
     *
     * @param {object} state - {x, y, vx, vy}
     * @returns {number} Specific orbital energy [J/kg]
     */
    energy(state) {
        const vSq = state.vx * state.vx + state.vy * state.vy;
        const r = Math.sqrt(state.x * state.x + state.y * state.y);
        return 0.5 * vSq - this.mu / r;
    }

    /**
     * Compute specific angular momentum (z-component).
     * L = x*vy - y*vx
     *
     * @param {object} state - {x, y, vx, vy}
     * @returns {number} Angular momentum [m^2/s]
     */
    angularMomentum(state) {
        return state.x * state.vy - state.y * state.vx;
    }
}
